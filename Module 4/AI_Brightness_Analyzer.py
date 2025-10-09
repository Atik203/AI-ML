nums = [int(x) for x in input().split()]

avg_brightness = sum(nums) / len(nums)

if avg_brightness > 170:
    print("Bright Image")
elif avg_brightness >= 85 and avg_brightness <= 170:
    print("Normal Image")
else:
    print("Dark Image")
