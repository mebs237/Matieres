        .data
ent1:.asciiz "entrer le premier  entier:\n "
ent2:.asciiz "entrer le second entier: \n "
resl:.asciiz "la somme des deux entiers donne:\n "

        .text  
main: 

       la $a0, ent1
       li $v0, 4
       syscall
       li $v0, 5
       syscall
       move $t0, $v0

       la $a0, ent2
       li $v0, 4
       syscall
       li $v0, 5
       syscall
       move $t1, $v0
       
       add $t2, $t0, $t1

       la $a0, resl
       li $v0, 4
       syscall


       move $a0, $t2
       li $v0, 1
       syscall
       jr $ra 
    