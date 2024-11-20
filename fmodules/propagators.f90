subroutine update_position_velocity_verlet(position_ad, velocity_ad, force_ad, mass_a, dt, n_atoms)
    implicit none
    
    !f2py intent(in) n_atoms
    integer, intent(in) :: n_atoms
    !f2py intent(inout) position_ad
    real*8, dimension(n_atoms, 3), intent(inout) :: position_ad
    !f2py intent(in) velocity_ad
    real*8, dimension(n_atoms, 3), intent(in) :: velocity_ad
    !f2py intent(in) force_ad
    real*8, dimension(n_atoms, 3), intent(in) :: force_ad
    !f2py intent(in) mass_a
    real*8, dimension(n_atoms), intent(in) :: mass_a
    !f2py intent(in) dt
    real*8, intent(in) :: dt

    integer a, d
    do a = 1, n_atoms
        do d = 1, 3
            position_ad(a,d) = position_ad(a,d) + velocity_ad(a,d)*dt + 0.5*force_ad(a,d)/mass_a(a)*dt**2
        end do
    end do
end subroutine update_position_velocity_verlet

subroutine update_velocity_velocity_verlet(velocity_ad, force_ad, mass_a, dt, n_atoms)
    implicit none
    
    !f2py intent(in) n_atoms
    integer, intent(in) :: n_atoms
    !f2py intent(inout) velocity_ad
    real*8, dimension(n_atoms, 3), intent(inout) :: velocity_ad
    !f2py intent(in) force_ad
    real*8, dimension(n_atoms, 3), intent(in) :: force_ad
    !f2py intent(in) mass_a
    real*8, dimension(n_atoms), intent(in) :: mass_a
    !f2py intent(in) dt
    real*8, intent(in) :: dt

    integer a, d
    do a = 1, n_atoms
        do d = 1, 3
            velocity_ad(a,d) = velocity_ad(a,d) + 0.5*force_ad(a,d)/mass_a(a)*dt
        end do
    end do
end subroutine update_velocity_velocity_verlet

subroutine get_state_coeff_values(coeff_in_s, coeff_out_s, nac_dt_ss, energy_s, n_states)
    implicit none
    
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(in) nac_dt_ss
    real*8, dimension(n_states, n_states), intent(in) :: nac_dt_ss
    !f2py intent(in) energy_s
    real*8, dimension(n_states), intent(in) :: energy_s
    !f2py intent(in) coeff_in_s
    complex*16, dimension(n_states), intent(in) :: coeff_in_s
    !f2py intent(out) coeff_out_s
    complex*16, dimension(n_states), intent(out) :: coeff_out_s

    integer s
    complex*16 :: ii = (0,1)
    do s = 1, n_states
        coeff_out_s(s) = -ii*energy_s(s)*coeff_in_s(s) - sum(nac_dt_ss(s,:)*coeff_in_s(:))
    end do
end subroutine 

subroutine state_coeff_rk4(state_coeff_s, nac_dt_ss, energy_s, dtq, n_states)
    implicit none
    
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(in) nac_dt_ss
    real*8, dimension(n_states, n_states), intent(in) :: nac_dt_ss
    !f2py intent(in) energy_s
    real*8, dimension(n_states), intent(in) :: energy_s
    !f2py intent(inout) coeff_in_s
    complex*16, dimension(n_states), intent(inout) :: state_coeff_s
    !f2py intent(in) dtq
    real*8, intent(in) :: dtq

    complex*16, dimension(n_states) :: k1, k2, k3, k4, temp
    call get_state_coeff_values(state_coeff_s, k1, nac_dt_ss, energy_s, n_states)
    temp = state_coeff_s + dtq*k1/2
    call get_state_coeff_values(temp, k2, nac_dt_ss, energy_s, n_states)
    temp = state_coeff_s + dtq*k2/2
    call get_state_coeff_values(temp, k3, nac_dt_ss, energy_s, n_states)
    temp = state_coeff_s + dtq*k3
    call get_state_coeff_values(temp, k4, nac_dt_ss, energy_s, n_states)
    state_coeff_s = state_coeff_s + dtq/6*(k1 + 2*k2 + 2*k3 + k4)
end subroutine state_coeff_rk4

subroutine update_tdc(velocity_ad, nac_dr_ssad, nac_dt_ss, n_states, n_atoms)
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(in) n_atoms
    integer, intent(in) :: n_atoms
    !f2py intent(in) velocity_ad
    real*8, intent(in), dimension(n_atoms, 3) :: velocity_ad
    !f2py intent(in) nac_dr_ssad
    real*8, intent(in), dimension(n_states, n_states, n_atoms, 3) :: nac_dr_ssad
    !f2py intent(out) nac_dt_ss
    real*8, intent(out), dimension(n_states, n_states) :: nac_dt_ss

    integer s1, s2
    do s1 = 1, n_states
        do s2 = 1, n_states
            if (s1 .ne. s2) then
                nac_dt_ss(s1,s2) = sum(velocity_ad(:,:) * nac_dr_ssad(s1,s2,:,:))
            else
                nac_dt_ss(s1,s2) = 0.
            end if
        end do
    end do
end subroutine update_tdc

subroutine update_overlap(state_old_ss, state_new_ss, overlap_ss, n_states)
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(in) state_old_ss
    real*8, intent(in), dimension(n_states, n_states) :: state_old_ss
    !f2py intent(in) state_new_ss
    real*8, intent(in), dimension(n_states, n_states) :: state_new_ss
    !f2py intent(out) overlap_ss
    real*8, intent(out), dimension(n_states, n_states) :: overlap_ss
    
    integer s1, s2
    do s1 = 1, n_states
        do s2 = 1, n_states
            overlap_ss(s1, s2) = sum(state_old_ss(:,s1) * state_new_ss(:,s2))
        end do
    end do
end subroutine update_overlap

subroutine normalise_state_coeff(state_coeff_s, n_states)
    !f2py intent(in) n_states
    integer, intent(in) :: n_states
    !f2py intent(inout) state_coeff_s
    complex*16, intent(inout), dimension(n_states) :: state_coeff_s

    real*8 norm

    norm = sum(abs(state_coeff_s)**2)
    state_coeff_s = state_coeff_s/norm
end subroutine normalise_state_coeff