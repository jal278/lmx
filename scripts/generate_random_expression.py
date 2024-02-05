'''
Utility for sampling benchmark expressions and converting into gplearn format.
'''

import numpy as np

from sympy.parsing import parse_expr

# List of all benchmark expressions
BENCHMARK_EXPRESSIONS = [
 '0.3*x1*sin(2*3.14*x1)',
 'pow(x1,3)*exp(-x1)*cos(x1)*sin(x1)*(pow(sin(x1),2)*cos(x1)-1)',
 'div(30*x1*x3,(x1-10)*pow(x2,2))',
 'div(x1*(x1+1),2)',
 'log(x1)',
 'sqrt(x1)',
 'log(x1+sqrt(pow(x1,2)+1))',
 'pow(x1,x2)',
 'x1*x2+sin((x1-1)*(x2-1))',
 'pow(x1,4)-pow(x1,3)+div(pow(x2,2),2)-x2',
 '6*sin(x1)*cos(x2)',
 'div(8,2+pow(x1,2)+pow(x2,2))',
 'div(pow(x1,3),5)+div(pow(x2,3),2)-x2-x1',
 '1.57+24.3*x4',
 '0.23+14.2*div((x4+x2),(3*x5))',
 '4.9*div((x4-x1+div(x2,x5)),(3*x5))-5.41',
 '0.13*sin(x3)-2.3',
 '3+2.13*log(abs(x5))',
 '1.3+0.13*sqrt(abs(x1))',
 '2.1380940889*(1-exp(-0.54723748542*x1))',
 '6.87+11*sqrt(abs(7.23*x1*x4*x5))',
 'div(sqrt(abs(x1)),log(abs(x2)))*div(exp(x3),pow(x4,2))',
 '0.81+24.3*div(2*x2+3*pow(x3,2),((4*pow(x4,3)+5*pow(x5,4))))',
 '6.87+11*cos(7.23*pow(x1,3))',
 '2-2.1*cos(9.8*x1)*sin(1.3*x5)',
 '32.0-(3.0*((tan(x1)/tan(x2))*(tan(x3)/tan(x4))))',
 '22.0-(4.2*((cos(x1)-tan(x2))*(tanh(x3)/sin(x4))))',
 '12.0-(6.0*((tan(x1)/exp(x2))*(log(x3)-tan(x4))))',
 'pow(x1,5)-2*pow(x1,3)+x1',
 'pow(x1,6)-2*pow(x1,4)+pow(x1,2)',
 'div(pow(x1,2)*pow(x2,2),(x1+x2))',
 'div(pow(x1,5),pow(x2,3))',
 'pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'sin(pow(x1,2))*cos(x1)-1',
 'sin(x1)+sin(x1+pow(x1,2))',
 'log(x1+1)+log(pow(x1,2)+1)',
 'sin(x1)+sin(pow(x2,2))',
 '2*sin(x1)*cos(x2)',
 'pow(x1,x2)',
'pow(x1,4)-pow(x1,3)+div(pow(x2,2),2)-x2',
 'pow(x1,4)-pow(x1,3)+div(pow(x2,2),2)-x2',
 '3.39*pow(x1,3)+2.12*pow(x1,2)+1.78*x1',
 'sin(pow(x1,2))*cos(x1)-0.75',
 'sin(1.5*x1)*cos(0.5*x2)',
 '2.7*pow(x1,x2)',
 'sqrt(1.23*x1)',
 'pow(x1,0.426)',
 '2*sin(1.3*x1)*cos(x2)',
 'log(x1+1.4)+log(pow(x1,2)+1.3)',
 '1./3+x1+sin(pow(x1,2))',
 'sin(pow(x1,2))*cos(x1)-2',
 'sin(pow(x1,3))*cos(pow(x1,2))-1',
 'log(x1+1)+log(pow(x1,2)+1)+log(x1)',
 'pow(x1,4)-pow(x1,3)+pow(x2,2)-x2',
 '4*pow(x1,4)+3*pow(x1,3)+2*pow(x1,2)+x1',
 'div(exp(x1)-exp(-1*x1),2)',
 'div(exp(x1)+exp(-1*x1),2)',
 'pow(x1,9)+pow(x1,8)+pow(x1,7)+pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 '6*sin(x1)*cos(x2)',
 'div(pow(x1,2)*pow(x2,2),(x1+x2))',
 'div(pow(x1,5),pow(x2,3))',
 'pow(x1,1/3)',
 'pow(x1,3)+pow(x1,2)+x1+sin(x1)+sin(pow(x2,2))',
 'pow(x1,1/5)',
 'pow(x1,2/3)',
 '4*sin(x1)*cos(x2)',
 'sin(pow(x1,2))*cos(x1)-5',
 'pow(x1,5)+pow(x1,4)+pow(x1,2)+x1',
 'exp(-1*pow(x1,2))',
 'pow(x1,8)+pow(x1,7)+pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'exp(-0.5*pow(x1,2))',
 'div(1,(1+pow(x1,-4)))+div(1,(1+pow(x2,-4)))',
 'pow(x1,9)+pow(x1,8)+pow(x1,7)+pow(x1,6)+pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'x1*x2+x3*x4+x5*x6+x1*x7*x9+x3*x6*x8',
 'div(pow(x1+1,3),pow(x1,2)-x1+1)',
 'div((pow(x1,5)-3*pow(x1,3)+1),(pow(x1,2)+1))',
 'div((pow(x1,6)+pow(x1,5)),(pow(x1,4)+pow(x1,3)+pow(x1,2)+x1+1))',
 'div(pow(x1+1,3),pow(x1,2)-x1+1)',
 'div((pow(x1,5)-3*pow(x1,3)+1),(pow(x1,2)+1))',
 'div((pow(x1,6)+pow(x1,5)),(pow(x1,4)+pow(x1,3)+pow(x1,2)+x1+1))',
 'sin(x1)+sin(x1+pow(x1,2))',
 'div(exp(-pow(x1-1,2)),(1.2+pow((x2-2.5),2)))',
 'exp(-x1)*pow(x1,3)*cos(x1)*sin(x1)*(cos(x1)*pow(sin(x1),2)-1)',
 'exp(-x1)*pow(x1,3)*cos(x1)*sin(x1)*(cos(x1)*pow(sin(x1),2)-1)*(x2-5)',
 'div(10,(5+(pow((x1-3),2)+pow((x2-3),2)+pow((x3-3),2)+pow((x4-3),2)+pow((x5-3),2))))',
 '30*(x1-1)*div(x3-1,(x1-10)*pow(x2,2))',
 '6*sin(x1)*cos(x2)',
 '(x1-3)*(x2-3)+2*sin(x1-4)*(x2-4)',
 'div(pow((x1-3),4)+pow((x2-3),3)-(x2-3),pow((x2-2),4)+10)',
'2.5*pow(x1,4)-1.3*pow(x1,3)+0.5*pow(x2,2)-1.7*x2',
 '8.0*pow(x1,2)+8.0*pow(x2,3)-15.0',
 '0.2*pow(x1,3)+0.5*pow(x2,3)-1.2*x2-0.5*x1',
 '1.5*exp(x1)+5.0*cos(x2)',
 '6.0*sin(x1)*cos(x2)',
 '1.35*x1*x2+5.5*sin((x1-1.0)*(x2-1.0))',
 'pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'pow(x1,5)+pow(x1,4)+pow(x1,3)+pow(x1,2)+x1',
 'sin(pow(x1,2))*cos(x1)-1',
 'log(x1+1)+log(pow(x1,2)+1)',
 '2*sin(x1)*cos(x2)',
 '0.58 + log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)',
 '2-2.1*cos(9.8*x1)*sin(1.3*x2)',
 'div(exp(-pow(x1-1,2)),(1.2+pow((x2-2.5),2)))',
 'div(1,(1+pow(x1,-4)))+div(1,(1+pow(x2,-4)))',
 '1./3+x1+sin(pow(x1,2))',
 '3.14*x1*x1']


# Same method of generating an expression as in LMX experiment
def randomly_labeled_benchmark_expr(benchmark_expr_list, n_vars=2):
    random_expr = np.random.choice(benchmark_expr_list).copy()
    for source_var_idx in range(1, 11):
        target_var_idx = np.random.randint(1, n_vars + 1)
        random_expr = random_expr.replace(f'x{source_var_idx}', f'x{target_var_idx}')
    return random_expr


# Convert sympy expression from tree to prefix form
def expr_to_prefix(expr):
    args = expr.args
    if len(args) == 0:
        return [expr]
    elif len(args) == 1:
        return [expr.func] + expr_to_prefix(args[0])
    elif 'Pow' in str(expr.func):
        base = args[0]

        # Handle sqrt
        if args[1] == 1/2:
            return ['sqrt'] + expr_to_prefix(base)

        # Handle integer powers
        try:
            power = int(args[1])
            if float(args[1]) - int(args[1]) > 0:
                return ['error']
        except:
            return ['error']
        prefix = []
        if power < 0:
            prefix.append('div')
            prefix.append(1.0)
            power = abs(power)
        for i in range(power-1):
            prefix.append('mul')
        for i in range(power):
            prefix += expr_to_prefix(base)
        return prefix

    else:
        prefix = []
        for i in range(len(args) - 1):
            prefix.append(expr.func)
        for arg in args:
            prefix += expr_to_prefix(arg)
        return prefix

    return 'ERROR!' # This point should never be reached


# Convert abstract prefix form to gplearn format
def prefix_to_program(prefix_expr, source_program):

    function_map = {}
    for function in source_program.function_set:
        function_map[function.name] = function

    program = []
    valid_program = True
    for node in prefix_expr:
            node_str = str(node).lower()
            function_found = False
            for function_name in function_map:
                if function_name in node_str:
                    program.append(function_map[function_name])
                    function_found = True
                    break
            if not function_found:
                if node_str[0] == 'x':
                    program.append(int(node_str[1:]) - 1)
                else:
                    try:
                        program.append(float(node))
                    except:
                        valid_program = False

    return program, valid_program


def generate_random_expression(reference_program, benchmark_expressions=BENCHMARK_EXPRESSIONS):

    valid_program_found = False
    while not valid_program_found:
        random_expr_str = randomly_labeled_benchmark_expr(benchmark_expressions, n_vars=2)
        random_expr_sympy = parse_expr(random_expr_str.replace('div', 'DIV'))
        random_prefix = expr_to_prefix(random_expr_sympy)
        random_program, valid_program_found = prefix_to_program(random_prefix, reference_program)

    return random_program

