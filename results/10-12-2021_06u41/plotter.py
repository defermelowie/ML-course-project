import json as json
import matplotlib.pyplot as plt

# Load data
with open(f'./results.json', 'r') as fd:
    results = json.load(fd)

# Plot setup
fig, (train_plt, cv_plt) = plt.subplots(2, sharex=True)

train_plt.set(xlabel='Lambda', ylabel='Accuracy')
train_plt.grid(visible=True, which='major', axis='both')
train_plt.label_outer()

cv_plt.set(xlabel='Lambda', ylabel='Accuracy')
cv_plt.grid(visible=True, which='major', axis='both')
cv_plt.label_outer()

line_styles = ['-', '--', '-.', ':']

# Plot data
for i, result in enumerate(results['results']):
    print(f'Result {i}: {result}')

    cv_plt.plot(
        result['lambda'],
        result['cv_accuracy'],
        f'k{line_styles[i]}',
        label=f'Cross validation {result["cv"]}'
    )
    train_plt.plot(
        result['lambda'],
        result['train_accuracy'],
        f'k{line_styles[i]}',
        label=f'Training {result["cv"]}'
    )

# Plot setup
train_plt.legend()
cv_plt.legend()

fig.tight_layout()

# Save and show
plt.savefig(f'./accuracy_lambda.pdf')
plt.show()
