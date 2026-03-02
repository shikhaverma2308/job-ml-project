import pandas as pd

# load real jobs
df = pd.read_csv("data/real_jobs.csv")

# user input
user_input = input("Enter your skills: ").lower().split()

# combine all job titles
all_jobs = " ".join(df['title']).lower().split()

# find missing skills
missing = set(all_jobs) - set(user_input)

print("Suggested skills to learn:")
print(list(missing)[:10])