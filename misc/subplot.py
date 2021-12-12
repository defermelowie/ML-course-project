import matplotlib.pyplot as plt

results = [
    {
        'cv': '0',
        'lambda': [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2],
        'cv_accuracy': [(-i**2.0 + 81.0)/81.0 for i in range(0, 9)],
        'train_accuracy': [(-i**1.9 + 81.0)/81.0 for i in range(0, 9)]
    },
    {
        'cv': '1',
        'lambda': [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2],
        'cv_accuracy': [(-i**1.7 + 81.0)/81.0 for i in range(0, 9)],
        'train_accuracy': [(-i**1.4 + 81.0)/81.0 for i in range(0, 9)]
    }
]


# ------------------------------------------------------------------------

fig, (train_plt, cv_plt) = plt.subplots(2, sharex=True)

train_plt.set(xlabel='Lambda', ylabel='Iterations')
train_plt.grid(visible=True, which='both', axis='both')
train_plt.label_outer()

cv_plt.set(xlabel='Lambda', ylabel='Iterations')
cv_plt.grid(visible=True, which='both', axis='both')
cv_plt.label_outer()

line_styles = ['-', '--', '-.', ':']

for i, result in enumerate(results):
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

train_plt.legend()
cv_plt.legend()

fig.tight_layout()

plt.show()
