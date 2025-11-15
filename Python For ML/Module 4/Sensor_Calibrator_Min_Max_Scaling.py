input_num = input()
nums = input_num.split()
x = float(nums[0])
min_v = float(nums[1])
max_v = float(nums[2])

norm = (x - min_v) / (max_v - min_v)
print(f"{norm:.2f}")
