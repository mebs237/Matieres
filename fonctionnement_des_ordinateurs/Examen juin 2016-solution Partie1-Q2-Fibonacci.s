# Cours de Fonctionnement des Ordinateurs, M. Charlier, UMONS	
# Examen Juin 2016, Partie 1 Question 2
# Suite de Fibonacci récursive

fibo:    
    #vérifie si cas récursif ou cas de base
    li      $t3, 1
    bgt     $a0, $t3, fibo_rec  #a0 > 1
    #cas de base
    addi    $v0, $a0, 0     #a0 <= 1 donc vo = a0
    jr      $ra

fibo_rec:
    addi    $a0, $a0, -1    #n : n-1
    addi    $sp, $sp, -8    #on va stocker $ra et le résultat du premier appel sur la pile
                            #on prévoit Fibonacci pour que n ne soit pas modifié une fois l'appel fini
    sw      $ra, 0($sp)     #mémorise $ra

    jal     fibo

    addi    $a0, $a0, -1    #n : n-2 = (n-1-1)
    sw      $v0, 4($sp)     #mémorise le résultat du premier appel récursif

    jal fibo

    lw      $t0, 4($sp)     #le résultat du premier appel récursif et le place dans $t0
    add     $v0, $t0, $v0   #on additionne le résultat actuel au résultat du premier appel récursif
    addi    $a0, $a0, 2     #restaure la valeur de n : n= n-2+2

    lw      $ra, 0($sp)     #on récupère $ra
    addi    $sp, $sp, 8
    jr      $ra 


main:
    li      $a0, 6                  #paramètre n
    addiu   $sp, $sp, -4            #place $ra sur la pile
    sw      $ra, 0($sp)
    jal     fibo                    #appel Fibonacci
    lw      $ra, 0($sp)             #récupère $ra
    addiu   $sp, $sp, 4
    move    $a0, $v0                #affichage résultat
    li      $v0, 1                  #affiche int
    syscall
    jr      $ra
