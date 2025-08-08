def count(params):
    return sum(p.numel() for p in params)

def get_params(model, has_quantum=False):
    if has_quantum:
        quantum_params = []
        for cell in model.quanvlstm.cell_list:
            quantum_params += list(cell.quanv.encoder.parameters()) + list(cell.quanv.q_layer.parameters())

        quantum_params_id = set(map(id, quantum_params))
        classical_params = [p for p in model.parameters() if id(p) not in quantum_params_id]

        quantum_num = count(quantum_params)
        classical_num = count(classical_params)

        print(f"Classical parameters count: {classical_num}")
        print(f"Quantum parameters count: {quantum_num}")

        return classical_params, quantum_params
    else:
        classical_params = list(model.parameters())
        
        print(f"Classical parameters count: {count(classical_params)}")
        return classical_params
