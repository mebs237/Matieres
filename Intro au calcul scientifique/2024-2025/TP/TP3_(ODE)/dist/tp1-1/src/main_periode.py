from numpy import array
from periode import vitesse_critique , plot_solution , x_max_sim,periode

epsilon = [.3 , .7]
mus = [0.0 , 0.1]
values = [(e , m) for e in epsilon for m in mus]


print("| ε   |  μ   |v_crit|period|x_max_sim|message")
print("-" * 60)
for e , m in values:
    crit = vitesse_critique(epsilon=e , mu=m )
    v0_crit = crit.v_crit
    message = crit.message
    x_max_sim_value = x_max_sim(epsilon=e , mu=m , v0=v0_crit,t_sim=60)
    period = periode(epsilon=e , mu=m , t_max=60).period
    print(f"|{e:.2f} |{m:.2f} |{v0_crit:.2f} |{period:.2f}  |{x_max_sim_value:.2f}   |{message}")
    print("-" * 60)

    plot_solution(epsilon=e , mu=m , y0 = array([0,v0_crit]) ,t_sim=60 , num_points=500)
