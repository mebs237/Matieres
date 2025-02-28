main:
        li $a0, 1
        li $a1, 2
        li $a2, 3
        li $a3, 4
        addiu $sp, $sp,-12
        sw $ra, 8($sp)
        li $t0, 5
        sw $t0, 4($sp)
        li $t0, 6
        sw $t0, 0($sp)
        jal mul6
        lw $ra, 8($sp)
        addiu $sp, $sp, 12
        li $v0, 1
        syscall
        jr $ra

mul6:
        lw $v0, 4($sp)
        lw $t0, 0($sp)
        mult $v0, $t0
        mflo $v0
        mult $v0, $a0
        mflo $v0
        mult $v0, $a1
        mflo $v0
        mult $v0, $a2
        mflo $v0
        mult $v0, $a3
        mflo $a0
        jr $ra



