main:
    li $a0, 5
loop:
    blez $a0, end
    li $v0, 1
    syscall
    add $a0, $a0, -1
    j   loop
end:
    jr $ra
    