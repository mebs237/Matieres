main:
        li $a0, 12
        li $a1, 1
        li $a2, 0
        li $v0, 0
        addiu $sp, $sp, -4
        sw $ra, 0($sp)
        jal mul_rec
        lw $ra, 0($sp) 
        addiu $sp, $sp,4
        jr $ra

mul_rec:
        beq $a0, $zero, L1
        beq $a1, $zero, L1
        addu $v0, $v0,  $a1
        addiu $a0, $a0, -1
        addiu $sp, $sp, -4
        sw $ra, 0($sp)
        jal mul_rec
        lw $ra, 0($sp) 
        addiu $sp, $sp,4
        jr $ra
    

L1 :
        move $a0, $v0
        li $v0, 1
        syscall
        jr $ra
