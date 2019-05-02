# NOTE: The sitename and dataname corresponding to the observation are 'y' by default
# Any latents that are not population level
model_constants = {
    'arm.anova_radon_nopred': {
        'population_effects':{'mu_a', 'sigma_a', 'sigma_y'},
        'ylims':(1000, 5000),
        'ylims_zoomed':(1000, 1200)
        },
    'arm.anova_radon_nopred_chr': {
        'population_effects':{'sigma_a', 'sigma_y', 'mu_a'},
        'ylims':(1000, 5000),
        'ylims_zoomed':(1000, 1200)
        },
    'arm.congress': {
        'population_effects':{'beta', 'sigma'},
        'sitename':'vote_88',
        'dataname':'vote_88',
        'ylims':(1000, 5000),   # CHANGE!
        'ylims_zoomed':(1000, 1200) # CHANGE!
        },
    'arm.earnings_latin_square': {
        'population_effects':{"sigma_a1", "sigma_a2", "sigma_b1", "sigma_b2", "sigma_c", "sigma_d", "sigma_y", 'mu_a1', 'mu_a2', 'mu_b1', 'mu_b2', 'mu_c', 'mu_d'},
        'ylims':(800, 5000),
        'ylims_zoomed':(800, 5000)
        },
    'arm.earnings_latin_square_chr': {
        'population_effects':{"sigma_a1", "sigma_a2", "sigma_b1", "sigma_b2", "sigma_c", "sigma_d", "sigma_y", 'mu_a1', 'mu_a2', 'mu_b1', 'mu_b2', 'mu_c', 'mu_d'},
        'ylims':(800, 5000),
        'ylims_zoomed':(800, 5000)
        },
    'arm.earnings_vary_si': {
        'population_effects':{"sigma_a1", "sigma_a2", "sigma_y", "mu_a1", "mu_a2"},
        'sitename':'log_earn',
        'dataname':'log_earn',
        'ylims':(800, 5000), # CHANGE!
        'ylims_zoomed':(800, 5000) # CHANGE!
        },
    'arm.earnings_vary_si_chr': {
        'population_effects':{"sigma_a1", "sigma_a2", "sigma_y", "mu_a1", "mu_a2"},
        'sitename':'log_earn',
        'dataname':'log_earn',
        'ylims':(800, 5000), # CHANGE!
        'ylims_zoomed':(800, 5000) # CHANGE!
        },
    'arm.earnings1': {
        'population_effects':{"sigma", "beta"},
        'sitename':'earn_pos',
        'dataname':'earn_pos',
        'ylims':(800, 5000), # CHANGE!
        'ylims_zoomed':(800, 5000) # CHANGE!
        },
    'arm.earnings2': {
        'population_effects':{"sigma", "beta"},
        'sitename':'log_earnings',
        'dataname':'log_earnings',
        'ylims':(800, 5000), # CHANGE!
        'ylims_zoomed':(800, 5000) # CHANGE!
        },
    'arm.election88_ch14': {
        'population_effects':{'mu_a', 'sigma_a', 'b'},
        'ylims':(1200, 2000),
        'ylims_zoomed':(1200, 1400)
        },
    'arm.election88_ch19': {
        'population_effects':{'beta', 'mu_age', 'sigma_age', 'mu_edu', 'sigma_edu', 'mu_age_edu', 'sigma_age_edu', 'mu_region', 'sigma_region', 'b_v_prev'},
        'ylims':(1200, 2000), # CHANGE!
        'ylims_zoomed':(1200, 1400) # CHANGE!
        },
    'arm.electric': {
        'population_effects':{'beta', 'mu_a', 'sigma_a', 'sigma_y'},
        'ylims':(1200, 2000), # CHANGE!
        'ylims_zoomed':(1200, 1400) # CHANGE!
        },
    'arm.electric_1a': {
        'population_effects':set(),
        'ylims':(1200, 2000), # CHANGE!
        'ylims_zoomed':(1200, 1400) # CHANGE!
        },
    'arm.hiv': {
        'population_effects':{'mu_a1', 'sigma_a1', 'mu_a2', 'sigma_a2', 'sigma_y'},
        'ylims':(1200, 2000), # CHANGE!
        'ylims_zoomed':(1200, 1400) # CHANGE!
        },
    'arm.wells_dist': {
        'population_effects':{'beta'},
        'sitename':'switched',
        'dataname':'switched',
        'ylims':(2000, 7500),
        'ylims_zoomed':(2000, 2500)
        },
    'arm.wells_dae_inter_c': {
        'population_effects':{'beta'},
        'sitename':'switched',
        'dataname':'switched',
        'ylims':(1800, 4000),
        'ylims_zoomed':(1800, 2200)
        },
    'arm.radon_complete_pool': {
        'population_effects':{'beta', 'sigma'},
        'ylims':(1000, 4000),
        'ylims_zoomed':(1000, 1400)
        },
    'arm.radon_group': {
        'population_effects':{'beta', 'sigma', 'mu_alpha', 'sigma_alpha', 'mu_beta', 'sigma_beta'},
        'ylims':(1000, 4000),
        'ylims_zoomed':(1000, 1200),
        },
    'arm.radon_inter_vary': {
        'population_effects':{'beta', 'sigma_y', 'sigma_a', 'sigma_b', 'sigma_beta', 'mu_a', 'mu_b', 'mu_beta'},
        'ylims':(1000, 5000),
        'ylims_zoomed':(1000, 1300)
        },
}
