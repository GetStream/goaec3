package bands

// bsNaNS16Vector is the int16 a NaN sample becomes in the vectorized
// FloatS16ToS16 loop on x86: cvttps2dq gives 0x80000000 and packssdw
// saturates it to -32768.
const bsNaNS16Vector int16 = -32768
