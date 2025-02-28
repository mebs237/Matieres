        .data
tab:    .word  -1, 3, 4, 7, 10, 90, 1024
        .text
main:
        la $a0, tab
        li $a1, 7
        li $a2, -1
        li $t1, 0
        addiu $sp, $sp, -4
        sw $ra, 0($sp)
        jal find_int
        lw $ra, 0($sp) 
        addiu $sp, $sp,4
        jr $ra

find_int:

        lw $t0, 0($a0)
        bgt $t0, $a2, end1
        beq $t0, $a2, end2
        addi $t1, $t1, 1
        addiu $a0, $a0, 4
        j   find_int
end1:
        li $a0, -1
        li $v0, 1
        syscall
        jr $ra
end2:
        move $a0, $t1
        li $v0, 1
        syscall
        jr $ra


        
