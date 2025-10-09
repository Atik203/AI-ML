keyword = ["ai", "data", "model", "learn", "train", "neural"]
s = input()

keyword_count = 0

words = s.split()
for word in words:
    if word in keyword:
        keyword_count += 1

if keyword_count >= 2:
    print("AI Detected")
else:
    print("Not AI Related")
