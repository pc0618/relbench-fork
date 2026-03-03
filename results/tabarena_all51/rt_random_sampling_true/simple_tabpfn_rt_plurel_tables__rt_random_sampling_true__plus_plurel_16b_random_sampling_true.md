# Simple TabPFN/RT/PluRel Tables

HF source: `TabArena/benchmark_results` commit `dc93ce38be052f9264cee8bcb7a7418d52ddd4dc` via HF API `https://huggingface.co/api/datasets/TabArena/benchmark_results`.
`tabarena_tabpfn` uses HF method `TABPFNV2 (default)`.

RelBench TabPFN source: `results/tabarena_all51/tabpfn-gpu-sharded-20260222-055604/tabarena_tabpfn_all51_cuda_combined_final.csv` (ok folds only).

Notes:
- Binary table reports **AUROC** (higher is better).
- Regression table reports **R^2** (higher is better), computed per fold as `1 - RMSE^2 / Var(y_test)` and averaged across folds.
- All RT/PluRel columns in this file (`rt`, `plurel_4b`, `plurel_16b`, `rt_random_sampling_true`, `plurel_16b_random_sampling_true`) were run with `seq_len=128` (`sl=128` in each CSV row's `rt_config`).

## Binary Classification (AUROC)
| dataset | tabarena_tabpfn | relbench_tabpfn | rt | plurel_4b | plurel_16b | rt_random_sampling_true | plurel_16b_random_sampling_true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amazon-employee-access | 0.83919 | 0.812083 | 0.467768 | 0.476347 | 0.508321 | 0.471667 | 0.499921 |
| apsfailure | 0.98975 | 0.987013 | 0.350622 | 0.029191 | 0.736911 | 0.402456 | 0.635399 |
| bank-customer-churn | 0.871723 | 0.878175 | 0.578162 | 0.42196 | 0.539326 | 0.489858 | 0.568726 |
| bank-marketing | 0.726477 | 0.759218 | 0.542517 | 0.430784 | 0.583099 | 0.559834 | 0.524813 |
| bioresponse | 0.872831 | 0.846753 | 0.494776 | 0.505082 | 0.517581 | 0.492207 | 0.492745 |
| blood-transfusion-service-center | 0.754857 | 0.752906 | 0.426369 | 0.53807 | 0.55799 | 0.520927 | 0.644436 |
| churn | 0.927864 | 0.925184 | 0.52356 | 0.363789 | 0.583531 | 0.496894 | 0.542698 |
| coil2000-insurance-policies | 0.753296 | 0.755337 | 0.5098 | 0.438616 | 0.552438 | 0.528683 | 0.526408 |
| credit-card-clients-default | 0.765429 | 0.782395 | 0.491015 | 0.364784 | 0.491525 | 0.516356 | 0.569303 |
| credit-g | 0.776149 | 0.776463 | 0.49918 | 0.57032 | 0.454182 | 0.482453 | 0.516674 |
| customer-satisfaction-in-airline | 0.992522 |  | 0.548397 | 0.436743 | 0.529616 | 0.537929 | 0.521527 |
| diabetes | 0.844407 | 0.839175 | 0.427393 | 0.305682 | 0.693405 | 0.470794 | 0.677365 |
| diabetes130us | 0.630801 | 0.64197 | 0.447113 | 0.494785 | 0.558271 | 0.489351 | 0.493383 |
| e-commereshippingdata | 0.74433 | 0.747508 | 0.530873 | 0.473037 | 0.55796 | 0.496965 | 0.582558 |
| fitness-club | 0.821891 | 0.821397 | 0.548664 | 0.434498 | 0.601915 | 0.509583 | 0.694595 |
| givemesomecredit | 0.846354 |  | 0.519811 | 0.459038 | 0.45447 | 0.531915 | 0.535016 |
| hazelnut-spread-contaminant-detection | 0.987913 | 0.986356 | 0.670638 | 0.802324 | 0.55738 | 0.442614 | 0.633518 |
| heloc | 0.800914 | 0.801246 | 0.58916 | 0.52983 | 0.511286 | 0.498789 | 0.600668 |
| hr-analytics-job-change-of-data-scientists | 0.788776 | 0.80166 | 0.513517 | 0.392042 | 0.427921 | 0.464766 | 0.535986 |
| in-vehicle-coupon-recommendation | 0.788855 | 0.806087 | 0.489213 | 0.522907 | 0.559534 | 0.498931 | 0.501722 |
| is-this-a-good-customer | 0.745537 | 0.738575 | 0.530529 | 0.542267 | 0.485567 | 0.509984 | 0.501282 |
| jm1 | 0.732462 | 0.733791 | 0.581764 | 0.376361 | 0.621935 | 0.491287 | 0.610875 |
| kddcup09-appetency | 0.772493 | 0.808587 | 0.529164 | 0.470297 | 0.502401 | 0.489628 | 0.496866 |
| marketing-campaign | 0.915313 | 0.896562 | 0.3888 | 0.389175 | 0.642586 | 0.475937 | 0.56437 |
| naticusdroid | 0.983049 | 0.98458 | 0.513954 | 0.501788 | 0.547844 | 0.448508 | 0.514963 |
| online-shoppers-intention | 0.934238 |  | 0.37428 | 0.365753 | 0.563321 | 0.45876 | 0.623937 |
| polish-companies-bankruptcy | 0.95862 | 0.959498 | 0.505396 | 0.421093 | 0.282624 | 0.611909 | 0.558898 |
| qsar-biodeg | 0.93616 | 0.933982 | 0.598193 | 0.639327 | 0.374635 | 0.557968 | 0.589542 |
| seismic-bumps | 0.771606 | 0.776896 | 0.340073 | 0.293424 | 0.720483 | 0.334212 | 0.549341 |
| taiwanese-bankruptcy-prediction | 0.941957 | 0.940225 | 0.468798 | 0.369834 | 0.167112 | 0.533551 | 0.648195 |

## Regression (R^2)
| dataset | tabarena_tabpfn | relbench_tabpfn | rt | plurel_4b | plurel_16b | rt_random_sampling_true | plurel_16b_random_sampling_true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| airfoil-self-noise | 0.973405 | 0.964565 | -0.009941 | -0.007743 | -0.004701 | -0.034721 | 0.075007 |
| another-dataset-on-used-fiat-500 | 0.858627 | 0.854704 | 0.02455 | -0.005291 | -0.004688 | 0.014308 | 0.344288 |
| concrete-compressive-strength | 0.934082 | 0.923747 | -0.048733 | -0.011523 | -0.007686 | -0.06003 | 0.152048 |
| diamonds | 0.980998 | 0.98318 | -0.116209 | -0.030898 | -0.028734 | 0.013547 | 0.386799 |
| food-delivery-time | 0.298843 | 0.350992 | -0.029613 | -0.000948 | 0.002207 | 0.009258 | 0.052585 |
| healthcare-insurance-expenses | 0.848455 | 0.8084 | -0.054643 | -0.008882 | -0.005859 | -0.093503 | 0.008718 |
| houses | 0.834821 | 0.873756 | -0.115966 | -0.007478 | -0.004856 | -0.056054 | 0.077553 |
| miami-housing | 0.92702 | 0.925721 | 0.040846 | 0.007686 | 0.010542 | -0.015862 | 0.117966 |
| physiochemical-protein | 0.660442 | 0.712194 | -0.046124 | -0.013563 | -0.009954 | -0.023185 | 0.014869 |
| qsar-fish-toxicity | 0.641846 | 0.635435 | -0.093263 | -0.015734 | -0.014614 | -0.015213 | 0.202649 |
| qsar-tid-11 | 0.727704 | 0.701272 | 0.031589 | -0.004531 | -0.003436 | -0.439365 | -0.023421 |
| superconductivity | 0.920966 | 0.920756 | -0.052712 | -0.008475 | -0.018437 | -0.017688 | 0.047484 |
| wine-quality | 0.370218 | 0.371602 | 0.000733 | -0.0052 | -0.00217 | -0.043863 | 0.060928 |
