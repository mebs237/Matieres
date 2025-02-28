        .data
   d:   .asciiz "entrer un entier"
        .text
main:
        la $a0, d
        li $v0, 4
        syscall
        li $v0, 5
        syscall
        beq $v0, 0, L1
        loop
  L1:
        li $a0, 0
        li $v0, 1
        syscall
        jr $ra
loop:
        beq $v0, 1, L3
        divu $v0, 2
        mflo $t0
        mfhi $a0
        li $v0,1
        syscall
        move $v0, $t0
        j   loop
  L3:
        li $a0, 1
        li $v0, 1
        syscall
        jr $ra