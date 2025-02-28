        .data
msg1:   .asciiz "veuillez entrer un entier: \n"
msg2:   .asciiz "trop grand \n"
msg3:   .asciiz "valeur acceptée\n"
        .text

main:
        la $a0, msg1
        li $v0, 4
        syscall
        li $v0, 5
        syscall
        li $a1, 10
        bgt $v0, $a1, L1
        bgtz $v0, L2
        jr $ra
 L2:
        la $a0, msg3
        li $v0, 4
        syscall
        j main
L1:
        la $a0, msg2
        li $v0, 4
        syscall
        j   main



