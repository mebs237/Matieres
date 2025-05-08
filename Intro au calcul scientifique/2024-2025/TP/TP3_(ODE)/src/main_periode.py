from numpy import array
from periode import vitesse_critique , plot_solution , x_max_sim,periode

epsilon = [.3 , .7]
mus = [0.0 , 0.1]
values = [(e , m) for e in epsilon for m in mus]

for e , m in values:
    crit = vitesse_critique(epsilon=e , mu=m )
    v0_crit = crit['v_crit']
    message = crit['add_info']
    x_max_sim_value = x_max_sim(epsilon=e , mu=m , v0=v0_crit,t_sim=60)
    print(f"pour ε = {e:.2f} et μ = {m:.2f} ")
    print("-" * 15)
    if m == 0.0:
        period = periode(epsilon=e , mu=m , t_max=60)['period']
        print(f"Période : {period:.2f} ")
        print("-" * 15)
    print(f"vitesse critique : {v0_crit:.2f} ")
    print(f"{message}")
    print("-" * 15)

    print(f"x_max_sim : {x_max_sim_value:.2f}")
    print("-" * 70)
    plot_solution(epsilon=e , mu=m , y0 = array([0,v0_crit]) ,t_sim=60 , num_points=500)
