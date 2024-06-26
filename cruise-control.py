# cruise-control.py - Exemplo de controle de cruzeiro da FBS
# RMM, 16 de maio de 2019
#
# O sistema de controle de cruzeiro de um carro é um sistema de feedback comum encontrado
# encontrado na vida cotidiana. O sistema tenta manter uma velocidade constante na
# presença de distúrbios causados principalmente por mudanças na inclinação de uma
# estrada. O controlador compensa essas incógnitas medindo a velocidade
# do carro e ajustando o acelerador adequadamente.
#
# Este arquivo explora a dinâmica e o controle do sistema de controle de cruzeiro,
# seguindo o material apresentado em Feedback Systems por Astrom e Murray.
# É usado um modelo não linear completo da dinâmica do veículo, com leis de controle PI e
# e leis de controle de espaço de estado.  São mostrados diferentes métodos de construção de sistemas de controle
# são mostrados, todos usando a classe InputOutputSystem (e subclasses).


import numpy as np
import matplotlib.pyplot as plt
from math import pi
import control as ct

#
# Seção 4.1: Modelagem e controle do controle de cruzeiro
#

# Modelo do veículo: vehicle()
#
# Para desenvolver um modelo matemático, começamos com um equilíbrio de forças para
# a carroceria do carro. Seja v a velocidade do carro, m a massa total
# (incluindo passageiros), F a força gerada pelo contato das
# contato das rodas com a estrada e Fd a força de perturbação devido à gravidade,
# atrito e arrasto aerodinâmico.


def vehicle_update(t, x, u, params={}):
    """Dinâmica do veículo para o sistema de controle de cruzeiro.

    Parâmetros
    ----------
    x : matriz
         Estado do sistema: velocidade do carro em m/s
    u : matriz
         Entrada do sistema: [throttle, gear, road_slope], em que throttle é
         um valor flutuante entre 0 e 1, marcha é um número inteiro entre 1 e 5,
         e road_slope está em rad.

    Retorno
    -------
    flutuante
        Aceleração do veículo

    """

    from math import copysign, sin
    sign = lambda x: copysign(1, x)         # definir a função sign()
    
    # Configurar os parâmetros do sistema
    m = params.get('m', 1600.)
    g = params.get('g', 9.8)
    Cr = params.get('Cr', 0.01)
    Cd = params.get('Cd', 0.32)
    rho = params.get('rho', 1.3)
    A = params.get('A', 2.4)
    alpha = params.get(
        'alpha', [40, 25, 16, 12, 10])      # Relação de transmissão / raio da roda

    # Definir variáveis de estado e entradas do veículo
    v = x[0]                           # Velocidade do veículo
    throttle = np.clip(u[0], 0, 1)     # acelerador do veículo
    gear = u[1]                        # transmissão do veículo
    theta = u[2]                       # Inclinação da estrada

    # Força gerada pelo motor

    omega = alpha[int(gear)-1] * v      # Velocidade angular do motor
    F = alpha[int(gear)-1] * motor_torque(omega, params) * throttle

    # Forças de perturbação
    #
    # A força de perturbação Fd tem três componentes principais: Fg, as forças resultantes da
    # à gravidade; Fr, as forças devido ao atrito de rolamento; e Fa, o arrasto aerodinâmico.
    # arrasto aerodinâmico.
    # Considerando que a inclinação da estrada seja \theta (theta), a gravidade dá a
    # força Fg = m g sin \theta.
    
    Fg = m * g * sin(theta)

    # Um modelo simples de atrito de rodagem é Fr = m g Cr sgn(v), em que Cr é
    # o coeficiente de atrito de rodagem e sgn(v) é o sinal de v (+/- 1) ou
    # zero se v = 0.
    
    Fr = m * g * Cr * sign(v)

    # O arrasto aerodinâmico é proporcional ao quadrado da velocidade: Fa =
    # 1/\rho Cd A |v| v, em que \rho é a densidade do ar, Cd é o
    # coeficiente de arrasto aerodinâmico dependente da forma e A é a área frontal
    # do carro.

    Fa = 1/2 * rho * Cd * A * abs(v) * v
    
    # Aceleração final do carro
    Fd = Fg + Fr + Fa
    dv = (F - Fd) / m
    
    return dv

# Modelo do motor: motor_torque
#
# A força F é gerada pelo motor, cujo torque é proporcional à
# à taxa de injeção de combustível, que por sua vez é proporcional a um
# sinal de controle 0 <= u <= 1 que controla a posição do acelerador. O torque também
# depende da velocidade ômega do motor.
    
def motor_torque(omega, params={}):
    # Configurar os parâmetros do sistema
    Tm = params.get('Tm', 190.)             # Constante de torque do motor
    omega_m = params.get('omega_m', 420.)   # Velocidade angular de pico do motor
    beta = params.get('beta', 0.4)          # Pico de desligamento do motor

    return np.clip(Tm * (1 - beta * (omega/omega_m - 1)**2), 0, None)

# Definir o sistema de entrada/saída para o veículo
vehicle = ct.NonlinearIOSystem(
    vehicle_update, None, name='vehicle',
    inputs=('u', 'gear', 'theta'), outputs=('v'), states=('v'))

# Figura 1.11: Um sistema de feedback para controlar a velocidade de um veículo. Neste
# exemplo, a velocidade do veículo é medida e comparada com a
# velocidade desejada.  O controlador é um controlador PI representado como uma função
# de transferência.  No livro-texto, as simulações são feitas para sistemas LTI, mas
# aqui simulamos o sistema não linear completo.

# Construa um controlador PI com rolloff, como uma função de transferência
Kp = 0.5                        # Ganho proporcional
Ki = 0.1                        # Ganho integral
control_tf =ct.TransferFunction(
    [Kp, Ki], [1, 0.01*Ki/Kp], name='control', inputs='u', outputs='y')

# Construção do sistema de controle de malha fechada
# Entradas: vref, marcha, theta
# Saídas: v (velocidade do veículo)
cruise_tf = ct.InterconnectedSystem(
    (control_tf, vehicle), name='cruise',
    connections=[
        ['control.u', '-vehicle.v'],
        ['vehicle.u', 'control.y']],
    inplist=['control.u', 'vehicle.gear', 'vehicle.theta'],
    inputs=['vref', 'gear', 'theta'],
    outlist=['vehicle.v', 'vehicle.u'],
    outputs=['v', 'u'])

# Defina o tempo e os vetores de entrada
T = np.linspace(0, 25, 101)
vref = 20 * np.ones(T.shape)
gear = 4 * np.ones(T.shape)
theta0 = np.zeros(T.shape)

# Agora simule o efeito de uma subida em t = 5 segundos
plt.figure()
plt.suptitle('Response to change in road slope')
vel_axes = plt.subplot(2, 1, 1)
inp_axes = plt.subplot(2, 1, 2)
theta_hill = np.array([
    0 if t <= 5 else
    4./180. * pi * (t-5) if t <= 6 else
    4./180. * pi for t in T])

for m in (1200, 1600, 2000):
    # Calcule o estado de equilíbrio do sistema
    X0, U0 = ct.find_eqpt(
        cruise_tf, [0, vref[0]], [vref[0], gear[0], theta0[0]], 
        iu=[1, 2], y0=[vref[0], 0], iy=[0], params={'m': m})

    t, y = ct.input_output_response(
        cruise_tf, T, [vref, gear, theta_hill], X0, params={'m': m})

    # Desenhe a velocidade
    plt.sca(vel_axes)
    plt.plot(t, y[0])

    # Desenhe a entrada
    plt.sca(inp_axes)
    plt.plot(t, y[1])

# Adicionar rótulos aos gráficos
plt.sca(vel_axes)
plt.ylabel('Speed [m/s]')
plt.legend(['m = 1000 kg', 'm = 2000 kg', 'm = 3000 kg'], frameon=False)

plt.sca(inp_axes)
plt.ylabel('Throttle')
plt.xlabel('Time [s]')

# Figura 4.2: Curvas de torque de um motor de carro típico. O gráfico à
# esquerda mostra o torque gerado pelo motor como uma função
# da velocidade angular do motor, enquanto a curva à direita mostra o
# torque como uma função da velocidade do carro para diferentes marchas.

# Figura 4.2
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# (a) - curva de torque única em função de ômega
ax = axes[0]
omega = np.linspace(0, 700, 701)
ax.plot(omega, motor_torque(omega))
ax.set_xlabel(r'Angular velocity $\omega$ [rad/s]')
ax.set_ylabel('Torque $T$ [Nm]')
ax.grid(True, linestyle='dotted')

# (b) - Curvas de torque em diferentes marchas, em função da velocidade
ax = axes[1]
v = np.linspace(0, 70, 71)
alpha = [40, 25, 16, 12, 10]
for gear in range(5):
    omega = alpha[gear] * v
    T = motor_torque(omega)
    plt.plot(v, T, color='#1f77b4', linestyle='solid')

# Configurar os eixos e o estilo
ax.axis([0, 70, 100, 200])
ax.grid(True, linestyle='dotted')

# Adicionar rótulos
plt.text(11.5, 120, '$n$=1')
ax.text(24, 120, '$n$=2')
ax.text(42.5, 120, '$n$=3')
ax.text(58.5, 120, '$n$=4')
ax.text(58.5, 185, '$n$=5')
ax.set_xlabel('Velocity $v$ [m/s]')
ax.set_ylabel('Torque $T$ [Nm]')

plt.suptitle('Torque curves for typical car engine')
plt.tight_layout()
plt.show(block=False)

# Figura 4.3: Carro com controle de velocidade de cruzeiro encontrando uma estrada inclinada

# Modelo do controlador PI: control_pi()
#
# Adicionamos a esse modelo um controlador de feedback que tenta regular a
# velocidade do carro na presença de distúrbios. Usaremos um controlador
# controlador proporcional-integral


def pi_update(t, x, u, params={}):
    # Obter os parâmetros do controlador de que precisamos
    ki = params.get('ki', 0.1)
    kaw = params.get('kaw', 2)  # Ganho anti-windup

    # Atribuir variáveis para entradas e estados (para facilitar a leitura)
    v = u[0]                    # velocidade atual
    vref = u[1]                 # Velocidade de referência
    z = x[0]                    # Erro integrado

    # Calcule a saída nominal do controlador (necessária para o anti-windup)
    u_a = pi_output(t, x, u, params)

    # Calcule a compensação anti-windup (dimensione por ki para levar em conta a estrutura)
    u_aw = kaw/ki * (np.clip(u_a, 0, 1) - u_a) if ki != 0 else 0

    # O estado é o erro integrado, menos a compensação anti-windup
    return (vref - v) + u_aw

def pi_output(t, x, u, params={}):
    # Obter os parâmetros do controlador de que precisamos
    kp = params.get('kp', 0.5)
    ki = params.get('ki', 0.1)

    # Atribuir variáveis para entradas e estados (para facilitar a leitura)
    v = u[0]                    # velocidade atual
    vref = u[1]                 # Velocidade de referência
    z = x[0]                    # Erro integrado

    # Controlador PI
    return kp * (vref - v) + ki * z

control_pi = ct.NonlinearIOSystem(
    pi_update, pi_output, name='control',
    inputs=['v', 'vref'], outputs=['u'], states=['z'],
    params={'kp': 0.5, 'ki': 0.1})

# Criar o sistema de malha fechada
cruise_pi = ct.InterconnectedSystem(
    (vehicle, control_pi), name='cruise',
    connections=[
        ['vehicle.u', 'control.u'],
        ['control.v', 'vehicle.v']],
    inplist=['control.vref', 'vehicle.gear', 'vehicle.theta'],
    outlist=['control.u', 'vehicle.v'], outputs=['u', 'v'])

# A Figura 4.3b mostra a resposta do sistema de circuito fechado.  A figura mostra
# que, mesmo que a colina seja tão íngreme que o acelerador mude de 0,17 para
# quase a aceleração total, o maior erro de velocidade é inferior a 1 m/s, e a
# velocidade desejada é recuperada após 20 s.

# Defina uma função para criar um gráfico de controle de cruzeiro "padrão"
def cruise_plot(sys, t, y, label=None, t_hill=None, vref=20, antiwindup=False,
                linetype='b-', subplots=None, legend=None):
    if subplots is None:
        subplots = [None, None]
    # Descubra os limites e os índices do gráfico
    v_min = vref-1.2; v_max = vref+0.5; v_ind = sys.find_output('v')
    u_min = 0; u_max = 2 if antiwindup else 1; u_ind = sys.find_output('u')

    # Certifique-se de que os limites superior e inferior de v estejam corretos
    while max(y[v_ind]) > v_max: v_max += 1
    while min(y[v_ind]) < v_min: v_min -= 1

    # Criar matrizes para valores de retorno
    subplot_axes = list(subplots)

    # Perfil de velocidade
    if subplot_axes[0] is None:
        subplot_axes[0] = plt.subplot(2, 1, 1)
    else:
        plt.sca(subplots[0])
    plt.plot(t, y[v_ind], linetype)
    plt.plot(t, vref*np.ones(t.shape), 'k-')
    if t_hill:
        plt.axvline(t_hill, color='k', linestyle='--', label='t hill')
    plt.axis([0, t[-1], v_min, v_max])
    plt.xlabel('Time $t$ [s]')
    plt.ylabel('Velocity $v$ [m/s]')

    # Perfil de entrada comandada
    if subplot_axes[1] is None:
        subplot_axes[1] = plt.subplot(2, 1, 2)
    else:
        plt.sca(subplots[1])
    plt.plot(t, y[u_ind], 'r--' if antiwindup else linetype, label=label)
    # Perfil de entrada aplicado
    if antiwindup:
        # TODO: traçar o sinal real do processo?
        plt.plot(t, np.clip(y[u_ind], 0, 1), linetype, label='Applied')
    if t_hill:
        plt.axvline(t_hill, color='k', linestyle='--')
    if legend:
        plt.legend(frameon=False)
    plt.axis([0, t[-1], u_min, u_max])
    plt.xlabel('Time $t$ [s]')
    plt.ylabel('Throttle $u$')

    return subplot_axes

# Defina o tempo e os vetores de entrada
T = np.linspace(0, 30, 101)
vref = 20 * np.ones(T.shape)
gear = 4 * np.ones(T.shape)
theta0 = np.zeros(T.shape)

# Calcule a configuração do acelerador de equilíbrio para a velocidade desejada (resolva para x
# e u, considerando a marcha, a inclinação e a velocidade de saída desejada)
X0, U0, Y0 = ct.find_eqpt(
    cruise_pi, [vref[0], 0], [vref[0], gear[0], theta0[0]],
    y0=[0, vref[0]], iu=[1, 2], iy=[1], return_y=True)

# Agora simule o efeito de uma colina em t = 5 segundos
plt.figure()
plt.suptitle('Car with cruise control encountering sloping road')
theta_hill = [
    0 if t <= 5 else
    4./180. * pi * (t-5) if t <= 6 else
    4./180. * pi for t in T]
t, y = ct.input_output_response(cruise_pi, T, [vref, gear, theta_hill], X0)
cruise_plot(cruise_pi, t, y, t_hill=5)

#
# Exemplo 7.8: Realimentação do espaço de estado com ação integral
#

# Modelo de controlador de espaço de estado: control_sf_ia()
#
# Construa um controlador de espaço de estado com ação integral, linearizado em torno de
# um ponto de equilíbrio.  O controlador é construído em torno do ponto de equilíbrio
# (x_d, u_d) e inclui compensação de feedback e feedforward.
#
# Entradas do controlador: (x, y, r) estados do sistema, saída do sistema, referência
# Estado do controlador: erro integrado z (y - r)
# Saída do controlador: controle de feedback do estado u
#
# Observação: para tornar a estrutura do controlador mais clara, implementamos isso
# como um módulo de entrada/saída "não linear", embora o sistema real de entrada/saída
# seja linear.  Isso também permite o uso de parâmetros para definir o
# ponto de operação e ganhos para o controlador.

def sf_update(t, z, u, params={}):
    y, r = u[1], u[2]
    return y - r

def sf_output(t, z, u, params={}):
    # Obter os parâmetros do controlador de que precisamos
    K = params.get('K', 0)
    ki = params.get('ki', 0)
    kf = params.get('kf', 0)
    xd = params.get('xd', 0)
    yd = params.get('yd', 0)
    ud = params.get('ud', 0)

    # Obter o estado do sistema e a entrada de referência
    x, y, r = u[0], u[1], u[2]

    return ud - K * (x - xd) - ki * z + kf * (r - yd)

# Obter o estado do sistema e a entrada de referência
control_sf = ct.NonlinearIOSystem(
    sf_update, sf_output, name='control',
    inputs=('x', 'y', 'r'),
    outputs=('u'),
    states=('z'))

# Criar o sistema de loop fechado para o controlador de espaço de estado
cruise_sf = ct.InterconnectedSystem(
    (vehicle, control_sf), name='cruise',
    connections=[
        ['vehicle.u', 'control.u'],
        ['control.x', 'vehicle.v'],
        ['control.y', 'vehicle.v']],
    inplist=['control.r', 'vehicle.gear', 'vehicle.theta'],
    outlist=['control.u', 'vehicle.v'], outputs=['u', 'v'])

# Calcule a linearização da dinâmica em torno do ponto de equilíbrio

# Y0 representa o estado estável com controle PI => podemos usá-lo para
# Identificar a velocidade de estado estável e a configuração necessária do acelerador.
xd = Y0[1]
ud = Y0[0]
yd = Y0[1]

# Calcule o sistema linearizado no ponto de partida da eq
cruise_linearized = ct.linearize(vehicle, xd, [ud, gear[0], 0])

# Construa as matrizes de ganho para o sistema
A, B, C = cruise_linearized.A, cruise_linearized.B[0, 0], cruise_linearized.C
K = 0.5
kf = -1 / (C * np.linalg.inv(A - B * K) * B)

# Resposta do sistema sem termo de realimentação integral
plt.figure()
plt.suptitle('Cruise control with proportional and PI control')
theta_hill = [
    0 if t <= 8 else
    4./180. * pi * (t-8) if t <= 9 else
    4./180. * pi for t in T]
t, y = ct.input_output_response(
    cruise_sf, T, [vref, gear, theta_hill], [X0[0], 0],
    params={'K': K, 'kf': kf, 'ki': 0.0, 'kf': kf, 'xd': xd, 'ud': ud, 'yd': yd})
subplots = cruise_plot(cruise_sf, t, y, label='Proportional', linetype='b--')

# Resposta do sistema com realimentação de estado + ação integral
t, y = ct.input_output_response(
    cruise_sf, T, [vref, gear, theta_hill], [X0[0], 0],
    params={'K': K, 'kf': kf, 'ki': 0.1, 'kf': kf, 'xd': xd, 'ud': ud, 'yd': yd})
cruise_plot(cruise_sf, t, y, label='PI control', t_hill=8, linetype='b-',
            subplots=subplots, legend=True)

# Exemplo 11.5: simular o efeito de uma colina (mais íngreme) em t = 5 segundos
#
# O efeito windup ocorre quando um carro encontra uma colina tão íngreme (6
# graus) que o acelerador satura quando o controlador de velocidade de cruzeiro tenta
# manter a velocidade.

plt.figure()
plt.suptitle('Cruise control with integrator windup')
T = np.linspace(0, 70, 101)
vref = 20 * np.ones(T.shape)
theta_hill = [
    0 if t <= 5 else
    6./180. * pi * (t-5) if t <= 6 else
    6./180. * pi for t in T]
t, y = ct.input_output_response(
    cruise_pi, T, [vref, gear, theta_hill], X0,
    params={'kaw': 0})
cruise_plot(cruise_pi, t, y, label='Commanded', t_hill=5, antiwindup=True,
            legend=True)

# Exemplo 11.6: adicionar compensação anti-windup
#
# O anti-windup pode ser aplicado ao sistema para melhorar a resposta. Devido ao
# do feedback do modelo do atuador, a saída do integrador é
# rapidamente redefinida para um valor tal que a saída do controlador esteja no
# limite de saturação.

plt.figure()
plt.suptitle('Cruise control with integrator anti-windup protection')
t, y = ct.input_output_response(
    cruise_pi, T, [vref, gear, theta_hill], X0,
    params={'kaw': 2.})
cruise_plot(cruise_pi, t, y, label='Commanded', t_hill=5, antiwindup=True,
            legend=True)

# Se estiver sendo executado como um programa autônomo, mostre os gráficos e aguarde antes de fechar
import os
if __name__ == '__main__' and 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
else:
    plt.show(block=False)
