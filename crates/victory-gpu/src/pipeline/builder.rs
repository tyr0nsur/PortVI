//! The title's acceleration-structure builder runs whole or not at all.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another. A dispatch whose host pipeline is missing is
//! dropped on its own (`State::dispatch`), and some of the builder's are
//! missing for a long time or for good: refused (64-lane waves on a host of
//! 32-lane waves), not translated, or still being made. The stages that do
//! run then work on what the missing ones never wrote, and write the result
//! into title memory: on fxceless/victory-fork, garbage bounds that crashed
//! the title. So the builder's dispatches run only in frames where every
//! pipeline it has created has its host pipeline; otherwise none of them
//! run (the structures keep what they held). A dispatch of a made pipeline
//! can still be dropped (`State::bind` fails, say): that holds the builder
//! back at once, for the rest of the frame and while that pipeline lives.
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.
//!
//! Traversal programs wait for a completed producer command from an earlier
//! frame. A missing builder stage or a dropped dispatch invalidates the
//! producer state. A title without a registered builder supplies its own
//! structures.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        let next = base.checked_add(size).filter(|&end| end > base).map(|end| (base, end));
        let mut image = IMAGE.lock().unwrap_or_else(|e| e.into_inner());
        if *image != next {
            first_submission().invalidate();
            *image = next;
        }
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// First completed builder submission since the last mapping or dropped
/// dispatch. A callback queued before an invalidation belongs to its old epoch.
#[derive(Default)]
struct FirstSubmission {
    epoch: u64,
    frame: Option<u64>,
}

impl FirstSubmission {
    fn invalidate(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.frame = None;
    }

    fn completed(&mut self, epoch: u64, frame: u64) {
        if self.epoch == epoch {
            self.frame.get_or_insert(frame);
        }
    }

    fn available(&self, frame: u64) -> bool {
        self.frame.is_some_and(|completed| completed != frame)
    }
}

static FIRST_SUBMISSION: Mutex<FirstSubmission> =
    Mutex::new(FirstSubmission { epoch: 0, frame: None });

fn first_submission() -> std::sync::MutexGuard<'static, FirstSubmission> {
    FIRST_SUBMISSION.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record completion on the host queue only after the command was recorded.
pub(crate) fn submitted(list: &crate::host::HostList) {
    let epoch = {
        let first = first_submission();
        if first.frame.is_some() {
            return;
        }
        first.epoch
    };
    let frame = crate::frame::current();
    list.defer_after(move || first_submission().completed(epoch, frame));
}

pub(crate) fn built() -> bool {
    image().is_none() || first_submission().available(crate::frame::current())
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        let mut frames = [std::ptr::null_mut(); 24];
        // SAFETY: room for that many frames.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)]
            .iter()
            .map(|&f| f as usize)
            .collect()
    }
    #[cfg(not(windows))]
    Vec::new()
}

/// One of the builder's pipelines (alive while the title holds it), and
/// whether a dispatch of it was dropped though it is made.
struct Noted {
    pipeline: Weak<Pending>,
    dropped: bool,
}

/// The builder's pipelines.
static PIPELINES: Mutex<Vec<Noted>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once (no frame's wait budget holds it back).
pub(crate) fn note(p: &Arc<Pending>) {
    PIPELINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Noted {
            pipeline: Arc::downgrade(p),
            dropped: false,
        });
    make::begin(p);
}

/// A dispatch of the builder's pipeline `p` was not run on the host
/// (`why`): none of the builder's dispatches run for the rest of the frame.
/// If `p` is made, none run while it lives either; one not made yet holds
/// the builder back through [`runs`] until it is.
pub(crate) fn dropped(p: &Arc<Pending>, why: &str) {
    first_submission().invalidate();
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        decided.1 = false;
    }
    if make::state(p) != Some(true) {
        return;
    }
    let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(n) = pipelines
        .iter_mut()
        .find(|n| std::ptr::eq(n.pipeline.as_ptr(), Arc::as_ptr(p)) && !n.dropped)
    {
        n.dropped = true;
        tracing::warn!(
            frame,
            why,
            "BVH builder held back: a dispatch of it was dropped"
        );
    }
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    static SAID: AtomicBool = AtomicBool::new(false);
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let (states, dropped): (Vec<Option<bool>>, usize) = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|n| n.pipeline.strong_count() > 0);
        let states = pipelines
            .iter()
            .filter_map(|n| n.pipeline.upgrade())
            .map(|p| make::state(&p))
            .collect();
        (states, pipelines.iter().filter(|n| n.dropped).count())
    };
    let open = all_made(&states) && dropped == 0;
    if open != decided.1 || !SAID.swap(true, Ordering::Relaxed) {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            dropped,
            "BVH builder {}",
            if open {
                "runs"
            } else if dropped > 0 {
                "held back: a dispatch of it was dropped"
            } else {
                "held back: not all its pipelines are made"
            }
        );
    }
    *decided = (frame, open);
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    forget_decision();
    first_submission().invalidate();
    guard
}

/// As if a new frame began: [`runs`] decides afresh.
#[cfg(test)]
pub(crate) fn forget_decision() {
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversals_wait_for_completed_work_and_ignore_stale_callbacks() {
        let mut first = FirstSubmission::default();
        assert!(!first.available(5));
        let epoch = first.epoch;
        first.completed(epoch, 5);
        assert!(!first.available(5));
        assert!(first.available(6));
        first.invalidate();
        first.completed(epoch, 6);
        assert!(!first.available(7));
        first.completed(first.epoch, 7);
        assert!(first.available(8));
    }

    /// The builder runs only when every one of its pipelines is made: one
    /// refused or still being made holds all of them back.
    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[]), "no builder: nothing to hold back");
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        // A range that holds this test's own code (a caller of the check).
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }
}
