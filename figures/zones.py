import matplotlib.pyplot as plt

plt.figure(tight_layout=True)
ax = plt.gca()
for y in [0, 1, 2]:
    for i in range(-5, 5):
        r = plt.Rectangle((3 * i + .5 + y, 2 - y), width=2, height=.6,
                          facecolor='#777777')

        r2 = plt.Rectangle((3 * i + 2.5 + y, 2 - y), width=1, height=.6,
                           facecolor='white')

        ax.add_patch(r)
        ax.add_patch(r2)

plt.annotate('Forest 1', (5, 2.7), ha='center')
plt.annotate('Forest 2', (5, 1.7), ha='center')
plt.annotate('Forest 3', (5, .7), ha='center')

plt.xlim(0, 10)
plt.ylim(-.1, 3)
plt.yticks([])

plt.gca().xaxis.set_ticks_position('bottom')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Longitude")

plt.savefig('zones.eps')
