subroutine get_energies_and_states(H_ss, energy_s, state_ss, n_states)
    implicit none
    
    external :: dsyev
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(in) H_ss
    real*8, dimension(n_states, n_states), intent(in) :: H_ss
    !f2py intent(out) energy_s
    real*8, dimension(n_states), intent(out) :: energy_s
    !f2py intent(out) state_ss
    real*8, dimension(n_states, n_states), intent(out) :: state_ss

    real*8, dimension(n_states, n_states) :: A
    real*8, dimension(3*n_states) :: work
    integer :: lwork, info

    A = H_ss
    lwork = 3*n_states
    call dsyev('V', 'U', n_states, A, n_states, energy_s, work, lwork, info)
    state_ss = A
end subroutine get_energies_and_states

subroutine get_nac_and_gradient(ham_ss, state_ss, gradH_ssad, nac_dr_ssad, n_states, n_atoms)
    implicit none
    
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(in) n_atoms
    integer, intent(in) :: n_atoms
    !f2py intent(in) energy_s
    real*8, dimension(n_states, n_states), intent(in) :: ham_ss
    !f2py intent(in) state_ss
    complex (kind=8), dimension(n_states, n_states), intent(in) :: state_ss
    !f2py intent(in) gradH_ssad
    real (kind=8), dimension(n_states, n_states, n_atoms, 3), intent(in) :: gradH_ssad
    !f2py intent(out) nac_dr_ssad
    real (kind=8), dimension(n_states, n_states, n_atoms, 3), intent(out) :: nac_dr_ssad

    integer s1, s2, a, d
    nac_dr_ssad = 0.
    do s1 = 1, n_states
        do s2 = 1, n_states
            do a = 1, n_atoms
                do d = 1, 3
                    nac_dr_ssad(s1,s2,a,d) = sum(conjg(state_ss(:,s1))*matmul(gradH_ssad(:,:,a,d), state_ss(:,s2)))
                end do
            end do

            if (s1 .ne. s2) nac_dr_ssad(s1,s2,:,:) = nac_dr_ssad(s1,s2,:,:)/(ham_ss(s2,s2) - ham_ss(s1,s1))
        end do
    end do
end subroutine get_nac_and_gradient

subroutine spin_boson(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    ! V11(x) = ax**2 + bx
    ! V22(x) = ax**2 - bx
    ! V12(x) = c
    ! V21(x) = c
    real*8 aa, bb, cc, x
    aa = 0.1
    bb = 0.3
    cc = 0.1

    x = position_ad(1,1)
    H_ss(1,1) = aa*x**2 + bb*x
    H_ss(2,2) = aa*x**2 - bb*x
    H_ss(1,2) = cc
    H_ss(2,1) = cc

    gradH_ssad = 0.
    gradH_ssad(1,1,1,1) = 2*aa*x + bb
    gradH_ssad(2,2,1,1) = 2*aa*x - bb
end subroutine spin_boson

subroutine tully_1(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    ! V11(x) = a(1-exp(-bx)) if x > 0
    ! V11(x) = -a(1-exp(bx)) if x <= 0
    ! V22(x) = -V11(x)
    ! V12(x) = c exp(-dx**2)
    ! V21(x) = V12(x)
    real*8 aa, bb, cc, dd, x
    aa = 0.01
    bb = 1.6
    cc = 0.005
    dd = 1.0

    x = position_ad(1,1)
    if (x .gt. 0) then
        H_ss(1,1) = aa*(1 - exp(-bb*x))
    else
        H_ss(1,1) = -aa*(1 - exp(bb*x))
    end if
    H_ss(2,2) = -H_ss(1,1)
    H_ss(1,2) = cc*exp(-dd*x**2)
    H_ss(2,1) = H_ss(1,2)

    gradH_ssad = 0.
    if (x > 0) then
        gradH_ssad(1,1,1,1) = aa*bb*exp(-bb*x)
    else
        gradH_ssad(1,1,1,1) = aa*bb*exp(bb*x)
    end if
    gradH_ssad(2,2,1,1) = -gradH_ssad(1,1,1,1)
    gradH_ssad(1,2,1,1) = -2*cc*dd*x*exp(-dd*x**2)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)
end subroutine tully_1

subroutine tully_s(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    real*8 aa, bb, cc, x
    aa = 0.01
    bb = 1.6
    cc = 0.005

    x = position_ad(1,1)
    gradH_ssad = 0.

    H_ss(1,1) = aa*tanh(bb*x)
    gradH_ssad(1,1,1,1) = aa*bb*(1-tanh(bb*x)**2)
    H_ss(2,2) = -aa*tanh(bb*x)
    gradH_ssad(2,2,1,1) = -aa*bb*(1-tanh(bb*x)**2)

    H_ss(1,2) = cc*exp(-x**2)
    gradH_ssad(1,2,1,1) = -2*cc*x*exp(-x**2)
    H_ss(2,1) = H_ss(1,2)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)
end subroutine tully_s

subroutine tully_n(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(3,3), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(3,3,1,3), intent(out) :: gradH_ssad

    real*8 aa, bb, cc, w, x
    aa = 0.03
    bb = 1.6
    cc = 0.005
    w = 7

    x = position_ad(1,1)
    gradH_ssad = 0.

    H_ss(1,1) = aa*(g(x,0d0,bb) + g(x,w,bb) - g(x,3*w,bb) - g(x,4*w,bb) - 2)
    gradH_ssad(1,1,1,1) = aa*(dg(x,0d0,bb) + dg(x,w,bb) - dg(x,3*w,bb) - dg(x,4*w,bb))
    H_ss(2,2) = aa*(-g(x,0d0,bb) + g(x,2*w,bb) + g(x,3*w,bb) - g(x,5*w,bb))
    gradH_ssad(2,2,1,1) = aa*(-dg(x,0d0,bb) + dg(x,2*w,bb) + dg(x,3*w,bb) - dg(x,5*w,bb))
    H_ss(3,3) = aa*(-g(x,w,bb) - g(x,2*w,bb) + g(x,4*w,bb) + g(x,5*w,bb) + 2)
    gradH_ssad(3,3,1,1) = aa*(-dg(x,w,bb) - dg(x,2*w,bb) + dg(x,4*w,bb) + dg(x,5*w,bb))

    H_ss(1,2) = cc*e(x,0d0) + cc*e(x,3*w)
    H_ss(2,1) = H_ss(1,2)
    gradH_ssad(1,2,1,1) = cc*de(x,0d0) + cc*de(x,3*w)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)
    H_ss(1,3) = cc*e(x,w) + cc*e(x,4*w)
    H_ss(3,1) = H_ss(1,3)
    gradH_ssad(1,3,1,1) = cc*de(x,w) + cc*de(x,4*w)
    gradH_ssad(3,1,1,1) = gradH_ssad(1,3,1,1)
    H_ss(2,3) = cc*e(x,2*w) + cc*e(x,5*w)
    H_ss(3,2) = H_ss(2,3)
    gradH_ssad(2,3,1,1) = cc*de(x,2*w) + cc*de(x,5*w)
    gradH_ssad(3,2,1,1) = gradH_ssad(2,3,1,1)

contains
    real*8 function g(y, y0, k)
        real*8, intent(in) :: y, y0, k
        g = tanh(k*(y - y0))
    end function g

    real*8 function dg(y, y0, k) 
        real*8, intent(in) :: y, y0, k
        dg = k*(1 - tanh(k*(y - y0))**2)
    end function dg

    real*8 function e(y,y0)
        real*8, intent(in) :: y, y0
        e = exp(-(y - y0)**2)    
    end function e

    real*8 function de(y, y0)
        real*8, intent(in) :: y, y0
        de = -2*(y - y0)*exp(-(y - y0)**2)
    end function de
end subroutine tully_n

subroutine tully_2(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    ! V11(x) = 0
    ! V22(x) = -a exp(-bx**2) + e0
    ! V12(x) = c exp(-dx**2)
    ! V21(x) = V12(x)
    real*8 aa, bb, cc, dd, e0, x
    aa = 0.1
    bb = 0.28
    cc = 0.015
    dd = 0.06
    e0 = 0.05

    x = position_ad(1,1)
    H_ss(1,1) = 0.
    H_ss(2,2) = -aa*exp(-bb*x**2) + e0
    H_ss(1,2) = cc*exp(-dd*x**2)
    H_ss(2,1) = H_ss(1,2)

    gradH_ssad = 0.
    gradH_ssad(2,2,1,1) = 2*aa*bb*x*exp(-bb*x**2)
    gradH_ssad(1,2,1,1) = -2*cc*dd*x*exp(-dd*x**2)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)
end subroutine tully_2

subroutine tully_3(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    ! V11(x) = A
    ! V22(x) = -A
    ! V12(x) = V21(x) = B exp(Cx) if x < 0
    ! V12(x) = V21(x) = B (2 - exp(-Cx)) if x >= 0
    ! A = 6e-4, B = 0.1, C = 0.9
    real*8 aa, bb, cc, x
    aa = 6d-4
    bb = 0.1
    cc = 0.9

    x = position_ad(1,1)
    H_ss(1,1) = aa
    H_ss(2,2) = -aa
    if (x .lt. 0) then
        H_ss(1,2) = bb*exp(cc*x)
    else
        H_ss(1,2) = bb*(2 - exp(-cc*x))
    end if
    H_ss(2,1) = H_ss(1,2)

    gradH_ssad = 0.
    if (x .lt. 0) then
        gradH_ssad(1,2,1,1) = bb*cc*exp(cc*x)
    else
        gradH_ssad(1,2,1,1) = bb*cc*exp(-cc*x)
    end if
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)
end subroutine tully_3

subroutine sub_x(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(3,3), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(3,3,1,3), intent(out) :: gradH_ssad

    real*8 aa, bb, cc, x
    aa = 0.03
    bb = 1.6
    cc = 0.005

    x = position_ad(1,1)
    gradH_ssad = 0.

    H_ss(1,1) = aa*(tanh(bb*x) + tanh(bb*(x+7)))
    gradH_ssad(1,1,1,1) = aa*bb*(1-tanh(bb*x)**2 + 1-tanh(bb*(x+7))**2)
    H_ss(2,2) = -aa*(tanh(bb*x) + tanh(bb*(x-7)))
    gradH_ssad(2,2,1,1) = -aa*bb*(1-tanh(bb*x)**2 + 1-tanh(bb*(x-7))**2)
    H_ss(3,3) = -aa*(tanh(bb*(x+7)) - tanh(bb*(x-7)))
    gradH_ssad(3,3,1,1) = -aa*bb*(1-tanh(bb*(x+7))**2 - (1-tanh(bb*(x-7))**2))

    H_ss(1,2) = cc*exp(-x**2)
    gradH_ssad(1,2,1,1) = -2*cc*x*exp(-x**2)
    H_ss(2,1) = H_ss(1,2)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)

    H_ss(1,3) = cc*exp(-(x+7)**2)
    gradH_ssad(1,3,1,1) = -2*cc*(x+7)*exp(-(x+7)**2)
    H_ss(3,1) = H_ss(1,3)
    gradH_ssad(3,1,1,1) = gradH_ssad(1,3,1,1)

    H_ss(2,3) = cc*exp(-(x-7)**2)
    gradH_ssad(2,3,1,1) = -2*cc*(x-7)*exp(-(x-7)**2)
    H_ss(3,2) = H_ss(2,3)
    gradH_ssad(3,2,1,1) = gradH_ssad(2,3,1,1)
end subroutine sub_x

subroutine sub_s(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(3,3), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(3,3,1,3), intent(out) :: gradH_ssad

    real*8 aa, bb, cc, dd, x
    aa = 0.015
    bb = 1.0
    cc = 0.005
    dd = 0.5

    x = position_ad(1,1)
    gradH_ssad = 0.

    H_ss(1,1) = aa*(tanh(bb*(x-7)) - tanh(bb*(x+7))) + aa
    gradH_ssad(1,1,1,1) = aa*bb*(1-tanh(bb*(x-7))**2 - (1-tanh(bb*(x+7))**2))
    H_ss(2,2) = -H_ss(1,1)
    gradH_ssad(2,2,1,1) = -gradH_ssad(1,1,1,1)
    H_ss(3,3) = 2*aa*tanh(dd*x)
    gradH_ssad(3,3,1,1) = 2*aa*dd*(1-tanh(dd*x)**2)

    H_ss(1,2) = cc*(exp(-(x-7)**2) + exp(-(x+7)**2))
    gradH_ssad(1,2,1,1) = -2*cc*((x-7)*exp(-(x-7)**2) + (x+7)*exp(-(x+7)**2))
    H_ss(2,1) = H_ss(1,2)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)

    H_ss(1,3) = cc*exp(-x**2)
    gradH_ssad(1,3,1,1) = -2*cc*x*exp(-x**2)
    H_ss(3,1) = H_ss(1,3)
    gradH_ssad(3,1,1,1) = gradH_ssad(1,3,1,1)

    H_ss(2,3) = cc*exp(-x**2)
    gradH_ssad(2,3,1,1) = -2*cc*x*exp(-x**2)
    H_ss(3,2) = H_ss(2,3)
    gradH_ssad(3,2,1,1) = gradH_ssad(2,3,1,1)
end subroutine sub_s

subroutine sub_2(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    real*8 aa, bb, cc, dd, ff, gg, w, x, y, z
    aa = 0.2
    bb = 0.6
    cc = 0.015
    dd = 0.3
    ff = aa/4
    gg = 0.3
    w = 2

    x = position_ad(1,1)
    y = position_ad(1,2)
    z = bb*(x-1) - w*sin(gg*y)
    gradH_ssad = 0.

    H_ss(1,1) = -ff*tanh(bb*x)
    gradH_ssad(1,1,1,1) = -bb*ff*(1-tanh(bb*x)**2)
    H_ss(2,2) = aa*tanh(z) + 3./4.*aa
    gradH_ssad(2,2,1,1) = aa*bb*(1-tanh(z)**2)
    gradH_ssad(2,2,1,2) = -aa*gg*w*(1-tanh(z)**2)*cos(gg*y)

    H_ss(1,2) = cc*exp(-dd*x**2)
    gradH_ssad(1,2,1,1) = -2*cc*dd*x*exp(-dd*x**2)
    H_ss(2,1) = H_ss(1,2)
    gradH_ssad(2,1,1,1) = gradH_ssad(1,2,1,1)
end subroutine sub_2

subroutine tully_1_nd(position_ad, H_ss, gradH_ssad)
    implicit none
    
    !f2py intent(in) position_ad
    real*8, dimension(1,3), intent(in) :: position_ad
    !f2py intent(out) H_ss
    real*8, dimension(2,2), intent(out) :: H_ss
    !f2py intent(out) gradH_ssad
    real*8, dimension(2,2,1,3), intent(out) :: gradH_ssad

    ! V11(x) = a(1-exp(-bx)) if x > 0
    ! V11(x) = -a(1-exp(bx)) if x <= 0
    ! V22(x) = -V11(x)
    ! V12(x) = c exp(-dx**2)
    ! V21(x) = V12(x)
    real*8 aa, bb, cc, dd, x
    real*8 r, r_hat(1,3)

    aa = 0.01
    bb = 1.6
    cc = 0.005
    dd = 1.0
    r = sqrt(sum(position_ad**2))
    r_hat = position_ad/r

    x = position_ad(1,1)
    if (x .gt. 0) then
        H_ss(1,1) = aa*(1 - exp(-bb*r))
    else
        H_ss(1,1) = -aa*(1 - exp(-bb*r))
    end if
    H_ss(2,2) = -H_ss(1,1)
    H_ss(1,2) = cc*exp(-dd*r**2)
    H_ss(2,1) = H_ss(1,2)

    gradH_ssad = 0.
    if (x > 0) then
        gradH_ssad(1,1,:,:) = aa*bb*exp(-bb*r)*r_hat
    else
        gradH_ssad(1,1,:,:) = -aa*bb*exp(-bb*r)*r_hat
    end if
    gradH_ssad(2,2,:,:) = -gradH_ssad(1,1,:,:)
    gradH_ssad(1,2,:,:) = -2*cc*dd*exp(-dd*r**2)*position_ad
    gradH_ssad(2,1,:,:) = gradH_ssad(1,2,:,:)
end subroutine tully_1_nd
