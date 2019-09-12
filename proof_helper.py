import sympy as sm

# generic (symetric) B-tensor
Bxx, Bxy, Bxz, Byy, Byz, Bzz = sm.symbols('Bxx, Bxy, Bxz, Byy, Byz, Bzz')
B = sm.Matrix([[Bxx, Bxy, Bxz],[Bxy, Byy, Byz],[Bxz, Byz, Bzz]])
print('generic (symmetric) B')
sm.pretty_print(B)
print('\n')

# generic GNL tensor
Lxx, Lxy, Lxz, Lyx, Lyy, Lyz, Lzx, Lzy, Lzz = sm.symbols('Lxx, Lxy, Lxz, Lyx, Lyy, Lyz, Lzx, Lzy, Lzz')
L = sm.Matrix([[Lxx, Lxy, Lxz], [Lyx, Lyy, Lyz], [Lzx, Lzy, Lzz]])
print('generic L')
sm.pretty_print(L)
print('\n')

# time variable
t = sm.symbols('t')

# generic q-vector timecourse
qx = sm.Function('qx')(t)
qy = sm.Function('qy')(t)
qz = sm.Function('qz')(t)
q = sm.Matrix([[qx],[qy],[qz]])
print('generic q(t)')
sm.pretty_print(q)
print('\n')

# different definition of the same generic B-tensor
print('B_other: same B but computed from q(t)')
B_other = sm.integrate(q*q.T, t)
sm.pretty_print(B_other)
print('\n')
print('B_other symmetric? {}'.format(B_other == B_other.T))
print('\n')
# i.e 
# B[0,0] == B_other[0,0]
# Bxx == Integral(qx(t)**2, t)
# ...


# resulting B-tensor after distortion generic B-tensor by generic GNL tensor
print('B_dist: b-tensor computed from L*q(t)')
B_dist = sm.expand(sm.integrate(sm.expand((L*q)*(q.T*L.T)), t))
sm.pretty_print(B_dist)
print('\n')
print('B_dist symmetric? {}'.format(B_dist == B_dist.T))
print('\n')
# it's disgusting but it's only expressed in terms of B_other (a.k.a. B) and L
# i.e
# B_dist[1,2] = Integral(Lyx*Lzx*qx(t)**2, t) + Integral(Lyy*Lzy*qy(t)**2, t) + Integral(Lyz*Lzz*qz(t)**2, t) + Integral(Lyx*Lzy*qx(t)*qy(t), t) + Integral(Lyx*Lzz*qx(t)*qz(t), t) + Integral(Lyy*Lzx*qx(t)*qy(t), t) + Integral(Lyy*Lzz*qy(t)*qz(t), t) + Integral(Lyz*Lzx*qx(t)*qz(t), t) + Integral(Lyz*Lzy*qy(t)*qz(t), t)
#             = Lyx*Lzx*Integral(qx(t)**2, t) + Lyy*Lzy*Integral(qy(t)**2, t) + Lyz*Lzz*Integral(qz(t)**2, t) + Lyx*Lzy*Integral(qx(t)*qy(t), t) + Lyx*Lzz*Integral(qx(t)*qz(t), t) + Lyy*Lzx*Integral(qx(t)*qy(t), t) + Lyy*Lzz*Integral(qy(t)*qz(t), t) + Lyz*Lzx*Integral(qx(t)*qz(t), t) + Lyz*Lzy*Integral(qy(t)*qz(t), t)
#             = Lyx*Lzx*B_other[0,0]          + Lyy*Lzy*B_other[1,1]          + Lyz*Lzz*B_other[2,2]          + Lyx*Lzy*B_other[0,1]             + Lyx*Lzz*B_other[0,2]             + Lyy*Lzx*B_other[0,1]             + Lyy*Lzz*B_other[1,2]             + Lyz*Lzx*B_other[0,2]             + Lyz*Lzy*B_other[1,2]
#             = Lyx*Lzx*Bxx                   + Lyy*Lzy*Byy                   + Lyz*Lzz*Bzz                   + Lyx*Lzy*Bxy                      + Lyx*Lzz*Bxz                      + Lyy*Lzx*Bxy                      + Lyy*Lzz*Byz                      + Lyz*Lzx*Bxz                      + Lyz*Lzy*Byz

print('B_dist is garbage-looking but once you pull out the L.. from the integrals, you just get sums of elements of B!')
print('Now term by term')
l = ['x', 'y', 'z']

for ix in range(3):
	for iy in range(ix, 3):
		print('B_dist_{}_{}'.format(l[ix], l[iy]))
		sm.pretty_print(B_dist[ix,iy])
		print('\n')




