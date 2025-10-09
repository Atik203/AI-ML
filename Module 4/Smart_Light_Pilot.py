input_num = input()
nums = input_num.split()
brightness = float(nums[0])
threshold = float(nums[1])

if brightness >= threshold:
    print("ON")
else:
    print("OFF")
