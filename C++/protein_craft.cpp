/*ProteinCraft - HP Model
----------------------------------------
What this program does
- Take an HP sequence (e.g., "HPPHHPH").
- Place the chain on a 2D square lattice as a self-avoiding walk (no overlaps).
- Uses a simple greedy heuristic during placement (with limited restarts) to favor hydrophobic (H) contacts.
- Computes the HP model energy: each non-consecutive H-H orthogonal contact = -1.
- Prints the lattice and the final energy.

*/