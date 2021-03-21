from colour import Color
import matplotlib.pyplot as plt
import qutip as qt

def plot_measurements(vector_alice_list, vector_bob_list):
    num = length(vector_alice_list)
    b = qt.Bloch()
    b.vector_color = list(map(lambda x: x.rgb, list(
        Color("red").range_to(Color("purple"), num))))
    for i in range(num):
        vector_alice = vector_alice_list[i]
        vector_bob = vector_bob_list[i]
        b.add_vectors(vector_alice)
        b.add_vectors(vector_bob)
    b.show()
    plt.show()