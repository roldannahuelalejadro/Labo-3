import numpy as np
import sympy as sp
import inspect
import ast

def numerica_a_simbolica(func):
    """
    Convierte una función numérica en una expresión simbólica de SymPy.
    Ojo esto solo funciona para las funciones "reales" donde
    result = f(x1,x2,...)  
    pues mapea el estado de la funcion y si es compatible remplaza las llamadas
    de numpy np a sp, nada que no se pueda hacer a mano en realidad
    """


    source = inspect.getsource(func)
    tree = ast.parse(source)
    func_def = tree.body[0]  
    arg_names = [arg.arg for arg in func_def.args.args]  
    
    # Crear variables simbólicas
    variables = sp.symbols(arg_names)
    
    # Encontrar la línea donde se define 'result'
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign):
            if any(target.id == "result" for target in stmt.targets):
                expr_code = ast.unparse(stmt.value)  # Extraer código de la expresión
                
                # Reemplazar funciones de NumPy por sus equivalentes en SymPy
                expr_code = expr_code.replace("np.", "sp.")  
                
                # Convertir a expresión simbólica
                expr_simb = eval(expr_code, {"sp": sp}, dict(zip(arg_names, variables)))
                
                return expr_simb

    raise ValueError("No se encontró una expresión válida en la función.")



def sp_propagacion_error(func, convertir=False):
    """
    Convierte una función numérica o simbólica en su expresión de propagación de errores.
    
    Parámetros:
    - func: función numérica o expresión simbólica.
    - convertir: si es True, convierte una función numérica en simbólica.
    """
    if convertir:
        f = numerica_a_simbolica(func)
    else:
        f = func  # Se asume que `func` ya es simbólica

    # Obtener variables simbólicas de la expresión
    variables = sorted(f.free_symbols, key=lambda s: str(s))
    N = len(variables)

    # Crear símbolos de error
    errores = [sp.Symbol(f'delta_{var}') for var in variables]

    # Calcular gradiente
    gradiente = [sp.diff(f, var) for var in variables]

    # Fórmula de propagación de errores
    error_propagado = sp.sqrt(sum((gradiente[i] * errores[i])**2 for i in range(N)))

    # Convertir a LaTeX
    latex_error = sp.latex(error_propagado)

    print("Fórmula de propagación de errores:", error_propagado)
    print("Fórmula en LaTeX:", latex_error)

    return error_propagado, latex_error

