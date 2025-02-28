#fait un si ?
if __name__ == '__main__':

	# Fonction a minimiser
	fc = lambda x,y: (3*x**2) + (x*y) + (5*y**2)
	# Calcul des dérivées partielles
	D_x = lambda x,y : 6*x  + y
	D_y = lambda x,y : 10*y + x

	# Initialisation des variables
	x = 10
	y = -13
	# Pas d'apprentissage
	lr = 0.1
	print (" *** Valeur initial avant DSG ***")
	print ("      Fc= %s" % (fc(x,y)))
	print ("\n *** Nouvelles valeurs calculées lors de l'entrainement *** ")

  # epoch : période de minimisation
	for epoch in range(0,20):
		# Calcul des gradients
		G_x = D_x(x,y)
		G_y = D_y(x,y)
		# Appliquer la descente de gradients
		x = x - lr*G_x
		y = y - lr*G_y

		# Vérifier la nouvelle valeur
		print ("Fc= %s" % (fc(x,y)))

	print ("")
	# Afficher les valeurs finales de x et y (poids)
	print (" *** valeurs x et y sélectionnées à l'issue de 20 epoch la DSG ***")
	print ("x = %s" % x)
	print ("y = %s" % y)
