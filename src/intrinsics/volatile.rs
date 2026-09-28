//! Codegen of the volatile load and store intrinsics.
//!
//! A volatile access, such as one to a memory-mapped device register, must be
//! performed exactly as written. Cranelift optimizes ordinary loads and stores:
//! it merges repeated loads, forwards stored values to later loads and removes
//! dead stores. Every load and store emitted here therefore carries the
//! `volatile` memory flag, which exempts it from all of that.

use cranelift_codegen::ir::immediates::Offset32;

use crate::prelude::*;

/// The memory flags of a volatile access.
///
/// `aligned` is true for `volatile_load`/`volatile_store`, whose pointer must
/// be aligned, and false for their `unaligned_` variants.
fn volatile_flags(aligned: bool) -> MemFlagsData {
    let flags = MemFlagsData::new().with_notrap().with_volatile();
    if aligned { flags.with_aligned() } else { flags }
}

fn pair_offset(b_offset: Size) -> Offset32 {
    Offset32::new(b_offset.bytes().try_into().unwrap())
}

/// Alignment of a copy's source or destination, as `emit_small_memory_copy`
/// takes it: a `u8`, so larger alignments are clamped to the largest that fits.
fn copy_align(layout: TyAndLayout<'_>, aligned: bool) -> u8 {
    if aligned { layout.align.bytes().try_into().unwrap_or(128) } else { 1 }
}

/// Reads the value at `ptr` into `ret` with volatile loads.
pub(super) fn codegen_volatile_load<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    ptr: Value,
    ret: CPlace<'tcx>,
    aligned: bool,
) {
    let layout = ret.layout();
    let ptr = Pointer::new(ptr);
    let flags = volatile_flags(aligned);

    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            let val = ptr.load(fx, scalar_to_clif_type(fx.tcx, scalar), flags);
            ret.write_cvalue(fx, CValue::by_val(val, layout));
        }
        BackendRepr::SimdVector { element, count } => {
            let ty = scalar_to_clif_type(fx.tcx, element).by(count.as_u32()).unwrap();
            let val = ptr.load(fx, ty, flags);
            ret.write_cvalue(fx, CValue::by_val(val, layout));
        }
        BackendRepr::ScalarPair { a, b, b_offset } => {
            let val_a = ptr.load(fx, scalar_to_clif_type(fx.tcx, a), flags);
            let val_b = ptr.offset(fx, pair_offset(b_offset)).load(
                fx,
                scalar_to_clif_type(fx.tcx, b),
                flags,
            );
            ret.write_cvalue(fx, CValue::by_val_pair(val_a, val_b, layout));
        }
        _ => {
            // Any other value lives in memory. Copy it into `ret` with volatile
            // loads, or with a single `memcpy` call if it is large; either way
            // the source is read exactly once.
            if layout.size == Size::ZERO {
                return;
            }
            let src = ptr.get_addr(fx);
            let dest = ret.to_ptr().get_addr(fx);
            fx.bcx.emit_small_memory_copy(
                fx.target_config,
                dest,
                src,
                layout.size.bytes(),
                copy_align(layout, true),
                copy_align(layout, aligned),
                true,
                flags,
            );
        }
    }
}

/// Writes `val` to `ptr` with volatile stores.
pub(super) fn codegen_volatile_store<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    ptr: Value,
    val: CValue<'tcx>,
    aligned: bool,
) {
    let layout = val.layout();
    let ptr = Pointer::new(ptr);
    let flags = volatile_flags(aligned);

    match layout.backend_repr {
        BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. } => {
            let val = val.load_scalar(fx);
            ptr.store(fx, val, flags);
        }
        BackendRepr::ScalarPair { a: _, b: _, b_offset } => {
            let (val_a, val_b) = val.load_scalar_pair(fx);
            ptr.store(fx, val_a, flags);
            ptr.offset(fx, pair_offset(b_offset)).store(fx, val_b, flags);
        }
        _ => {
            // Any other value lives in memory. Copy it to `ptr` with volatile
            // stores, or with a single `memcpy` call if it is large; either way
            // the destination is written exactly once.
            if layout.size == Size::ZERO {
                return;
            }
            let (src, meta) = val.force_stack(fx);
            assert!(meta.is_none());
            let src = src.get_addr(fx);
            let dest = ptr.get_addr(fx);
            fx.bcx.emit_small_memory_copy(
                fx.target_config,
                dest,
                src,
                layout.size.bytes(),
                copy_align(layout, aligned),
                copy_align(layout, true),
                true,
                flags,
            );
        }
    }
}
