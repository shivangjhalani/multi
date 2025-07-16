# Multimodal CoCoNuT Debug Analysis

## Critical Issues Identified

### 1. InternVL3 Forward Method Compatibility
**Error**: `'bool' object has no attribute 'sum'`
**Location**: InternVL3's `modeling_internvl_chat.py` line 126
**Root Cause**: The InternVL3 model expects specific parameter formats that our wrapper isn't providing correctly.

### 2. Visual Embedding Shape Mismatch  
**Error**: `The size of tensor a (108) must match the size of tensor b (128) at non-singleton dimension 1`
**Root Cause**: Mismatch between expected visual embedding dimensions and actual tensor shapes.

### 3. Unexpected Keyword Arguments
**Error**: `InternVLChatModel.forward() got an unexpected keyword argument 'num_patches_list'`
**Root Cause**: Our collator is passing parameters that InternVL3 doesn't expect.

## Debugging Strategy

### Phase 1: Fix InternVL3 Integration
1. **Examine InternVL3's expected forward signature**
2. **Fix parameter passing in our wrapper**
3. **Ensure proper visual embedding handling**

### Phase 2: Fix Data Pipeline Issues
1. **Remove unexpected parameters from collator**
2. **Fix tensor shape mismatches**
3. **Ensure proper multimodal batch formatting**

### Phase 3: Validate CoCoNuT Logic
1. **Test latent token detection**
2. **Verify continuous thought feedback**
3. **Validate multi-pass forward logic**

## Immediate Actions Required

1. **Inspect InternVL3 source code** to understand expected parameters
2. **Fix our MultimodalCoconut wrapper** to properly interface with InternVL3
3. **Update MultimodalCollator** to only pass expected parameters
4. **Add proper error handling** for shape mismatches
5. **Create minimal test cases** to validate each component

## Expected Fixes

### Fix 1: Update MultimodalCoconut._standard_multimodal_forward()
- Remove unsupported parameters before calling base_model
- Add proper error handling for parameter mismatches
- Ensure image_flags is properly formatted

### Fix 2: Update MultimodalCollator
- Remove num_patches_list from forward batch
- Store it separately for internal use only
- Ensure proper tensor formatting for InternVL3

### Fix 3: Fix Visual Embedding Integration
- Investigate InternVL3's expected visual embedding format
- Fix tensor shape mismatches in _prepare_multimodal_embeddings
- Add proper error handling for dimension mismatches

## Testing Plan

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test full pipeline with minimal data
4. **Regression Tests**: Ensure fixes don't break existing functionality