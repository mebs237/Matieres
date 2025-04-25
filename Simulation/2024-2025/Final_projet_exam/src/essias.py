from numpy.random import uniform
from Tests import Chi2Test
from Generators import OurGenerator , PythonGenerator
from Analysing import  analyse_sequence
seq = uniform(0,1,50)

ogen = OurGenerator(seq , name="seq_generator")
pgen = PythonGenerator()
seq_pgen = pgen.generate(60)
seq_ogen = ogen.generate(60)
reso = Chi2Test().test(seq_ogen)
resp = Chi2Test().test(seq_pgen)

resa = analyse_sequence(seq_ogen, granularities=1.0,name="seq_ogen")

resa.save_hist('data_results')

print(reso)
print(resp)
