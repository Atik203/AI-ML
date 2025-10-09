s = [x for x in input().split()]

a_count = 0
b_count = 0

for i in s:
    if i == "A":
        a_count += 1
    elif i == "B":
        b_count += 1

if a_count / len(s) > 0.7 or b_count / len(s) > 0.7:
    print("Biased Model")
else:
    print("Fair Model")
