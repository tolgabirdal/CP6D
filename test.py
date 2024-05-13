import json

to_plot_trans = {
    '111': [i for i in range(100)]
}
with open('to_plot_trans.json', 'w') as f:
    json.dump(to_plot_trans, f)