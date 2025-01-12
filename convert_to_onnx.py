import onnx
import onnxruntime
import torch

def rename_onnx_io(
    input_onnx_path: str,
    output_onnx_path: str,
    new_input_name: str = "pixel_values",
    new_output_name: str = "logits"
):
    # 1) Cargar modelo
    model = onnx.load(input_onnx_path)
    graph = model.graph

    # 2) Tomar los nombres originales del primer input y del primer output
    old_input_name = graph.input[0].name
    old_output_name = graph.output[0].name
    
    print("Nombres originales:")
    print(f"  Input:  {old_input_name}")
    print(f"  Output: {old_output_name}")

    # 3) Renombrar directamente en graph.input y graph.output
    graph.input[0].name = new_input_name
    graph.output[0].name = new_output_name

    # 4) Recorrer todos los nodos y actualizar referencias
    for node in graph.node:
        # Cambiar referencias al input antiguo
        node.input[:] = [
            new_input_name if x == old_input_name else x
            for x in node.input
        ]
        # Cambiar referencias al output antiguo
        node.output[:] = [
            new_output_name if x == old_output_name else x
            for x in node.output
        ]

    # 5) Guardar el modelo modificado
    onnx.save(model, output_onnx_path)
    print(f"\nModelo guardado como: {output_onnx_path}")

    # 6) Verificar que ONNXRuntime lo cargue correctamente
    session = onnxruntime.InferenceSession(output_onnx_path)
    print("Cargado correctamente con onnxruntime.\n")

    # Mostrar los nuevos nombres de input y output
    print("Nuevos nombres de entrada y salida:")
    for i, inp in enumerate(session.get_inputs()):
        print(f"  Input {i}:  name={inp.name}, shape={inp.shape}, type={inp.type}")
    for o, out in enumerate(session.get_outputs()):
        print(f"  Output {o}: name={out.name}, shape={out.shape}, type={out.type}")
    print("")

if __name__ == "__main__":
    # Nombre del ONNX original
    original_onnx_path = "model2.onnx"
    # Nombre del ONNX renombrado
    renamed_onnx_path = "model.onnx"

    # Llamar la función de renombre
    #rename_onnx_io(
    #    input_onnx_path=original_onnx_path,
    #    output_onnx_path=renamed_onnx_path,
    #    new_input_name="pixel_values",
    #    new_output_name="logits",
    #)

    # Prueba de inferencia (opcional)
    # --------------------------------
    # Cargar el modelo con onnxruntime:
    session = onnxruntime.InferenceSession("model.onnx")

    # Crear dummy_input para hacer inferencia rápida
    dummy_input = torch.randn(1, 3, 288, 288)
    
    # Nombre de la entrada en el modelo renombrado
    input_name = session.get_inputs()[0].name
    print(f"Nombre de la entrada: {input_name}")
    # Convertir tensor a numpy
    def to_numpy(tensor):
        return tensor.cpu().numpy()
    
    # Realizar la inferencia
    print("Realizando inferencia de prueba...")
    ort_outs = session.run(
        None,
        {input_name: to_numpy(dummy_input)}
    )
    print("¡Inferencia exitosa! Salida:")
    print(ort_outs)
