"""Quadratic four-region experiment; conventional approximation, no physics claim.

Reuses the competition reference's normalization, truncated arithmetic, Newton
step and uncomputation. Only the initial approximation and its QROM change.
"""
import dataclasses
import math
import numpy as np
import kickmix as km
import nr_reference_circuit as ref

CENTRED = (
    (1.125, (0.9428090415820635, -0.4214689832973239, 0.2812514007390986)),
    (1.375, (0.8528028654224415, -0.31131738445236035, 0.1699194116857967)),
    (1.625, (0.7844645405527362, -0.24204554408499326, 0.11176506832050342)),
    (1.875, (0.7302967433402214, -0.19515258668382723, 0.07808817767229422)),
)

def coefficient_table(prec):
    left = []
    for center, (a, b, c) in CENTRED:
        h = 0.125
        left.append((a - b*h + c*h*h, -b + 2*c*h, c))
    cases = left + [[v / (2**j * math.sqrt(2)) for j, v in enumerate(row)] for row in left]
    return [[round(v * 2**prec.coef_bits) for v in row] for row in cases]

def lookup_table(prec):
    table = coefficient_table(prec)
    packed = []
    for address in range(16):
        fourth, third, second, top = ((address >> i) & 1 for i in range(4))
        case = (4 + 2*second + third) if top else (2*third + fourth)
        packed.append(sum(v << (j*prec.coef_bits) for j, v in enumerate(table[case])))
    return packed

def horner(builder, inverse, prec, x, quarter, coefs, scratch):
    wc, width = prec.coef_bits, len(x)
    delta = [(quarter, 2)]
    delta += [(x[width-2-m], m) for m in range(3, min(prec.delta_bits, width-2)+1)]
    steps = [(coefs[1], coefs[2]), (coefs[0], coefs[1])]
    for acc, mult in (reversed(steps) if inverse else steps):
        for control, m in delta:
            terms = [(mult[i], i-m) for i in range(m, wc)]
            if terms:
                ref._add_row(builder, inverse, control, terms, acc, scratch, subtract=True)

def build(prec):
    n_p, width = 10, 22
    builder = km.CircuitBuilder()
    x = builder.create_quantum_register(width, name='x')
    flags = builder.alloc_qubits(n_p)
    quarter = builder.alloc_qubits(1)
    packed = builder.alloc_qubits(3*prec.coef_bits)
    coefs = [packed[j*prec.coef_bits:(j+1)*prec.coef_bits] for j in range(3)]
    square = builder.alloc_qubits(prec.square_bits)
    z = builder.alloc_qubits(prec.product_bits+2)
    y1 = builder.alloc_qubits(prec.output_bits(n_p))
    scratch = builder.alloc_qubits(max(prec.square_bits, prec.product_bits+2, prec.result_bits+1))
    address = km.array([x[-4], x[-3], x[-2], x[-1]])
    table = lookup_table(prec)
    p = y1[:len(y1)-n_p]

    for inverse in (False, True):
        if inverse:
            ref._phase_by_bits(builder, y1)
            ref._denormalize(builder, True, flags, y1)
            ref._multiply_y0(builder, True, coefs[0], z, p, scratch)
            ref._correct(builder, True, z)
            ref._multiply_m(builder, True, x, square, z, scratch)
            ref._square(builder, True, coefs[0], square, scratch)
            horner(builder, True, prec, x, quarter[0], coefs, scratch)
            builder.del_lookup(table=table, address=address, target=packed)
            ref._and(builder, True, x[-1], x[-4], quarter[0])
            ref._normalize(builder, True, x, flags)
        else:
            ref._normalize(builder, False, x, flags)
            ref._and(builder, False, x[-1], x[-4], quarter[0])
            builder.init_lookup(table=table, address=address, target=packed)
            horner(builder, False, prec, x, quarter[0], coefs, scratch)
            ref._square(builder, False, coefs[0], square, scratch)
            ref._multiply_m(builder, False, x, square, z, scratch)
            ref._correct(builder, False, z)
            ref._multiply_y0(builder, False, coefs[0], z, p, scratch)
            ref._denormalize(builder, False, flags, y1)
    for reg in (flags, quarter, packed, square, z, y1, scratch):
        builder.free(reg)
    return builder.finish_circuit()

def model(prec, xs):
    """Exact column-truncated integer model matching the emitted arithmetic."""
    xs = np.asarray(xs, dtype=np.int64)
    wc, wsq, wz, wp = prec.coef_bits, prec.square_bits, prec.product_bits, prec.result_bits
    s = (22 - (np.floor(np.log2(xs)).astype(np.int64)+1)) // 2
    xn = xs << (2*s)
    bit = lambda v, i: (v >> i) & 1
    top, second, third, fourth = (bit(xn, i) for i in (21,20,19,18))
    case = np.where(top, 4+2*second+third, 2*third+fourth)
    table = np.array(coefficient_table(prec), dtype=np.int64)
    c0,c1,c2 = [table[case,j].copy() for j in range(3)]
    delta = [(top & fourth, 2)] + [(bit(xn,20-m),m) for m in range(3,min(prec.delta_bits,20)+1)]
    def product(rows, operand, out_bits, operand_bits, width):
        acc = np.zeros_like(xs)
        for ctl, weight in rows:
            e = out_bits-operand_bits+weight
            acc += ctl * (operand << e if e >= 0 else operand >> -e)
        if ((acc < 0) | (acc >= 2**width)).any():
            raise OverflowError('product overflow')
        return acc
    for acc, mult in ((c1,c2),(c0,c1)):
        acc -= product([(b,-m) for b,m in delta],mult,wc,wc,wc)
        if ((acc<=0)|(acc>=2**wc)).any():
            raise OverflowError('Horner overflow')
    y = {k:bit(c0,wc-k) for k in range(1,wc+1)}
    sq = np.zeros_like(xs)
    for k in range(1,wc+1):
        if 2*k<=wsq:
            sq += y[k] << (wsq-2*k)
        for l in range(k+1,wc+1):
            if k+l-1<=wsq:
                sq += (y[k]&y[l]) << (wsq-k-l+1)
    z = product([(bit(xn,i),i-20) for i in range(22)],sq,wz,wsq,wz+2)
    correction = (3<<wz)-z
    if ((correction<0)|(correction>=2**(wz+2))).any():
        raise OverflowError('correction overflow')
    result = product([(y[k],-k) for k in range(1,wc+1)],correction,wp,wz,wp+1)
    return result << s
