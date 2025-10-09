n = int(input())
target = float(input())
total_loss = 0.0

for i in range(n):
    loss = float(input())
    total_loss += loss

average = total_loss / n

if average <= target:
    print("PASS")
else:
    print("RETRY")
