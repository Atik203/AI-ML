n = int(input())
yes_count = 0
no_count = 0

for i in range(n):
    vote = input()
    if vote == "YES":
        yes_count += 1
    elif vote == "NO":
        no_count += 1
if yes_count >= no_count:
    print("ACCEPT")
else:
    print("REJECT")
