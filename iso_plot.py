import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------------------------------------------
# 1) 실험 파라미터 (원하는 값으로 자유롭게 수정)
# --------------------------------------------------------------
lam = 633e-9          # 파장 (m)  –  He‑Ne 레이저 633 nm
a   = 30e-6           # 슬릿 폭 (m) – 30 µm
d   = 200e-6          # 슬릿 중심 간격 (m) – 200 µm
L   = 1.0             # 화면까지 거리 (m)

# 화면(관측면) 설정
screen_width = 0.02   # 화면 전체 가로 길이 (m) – 2 cm
N = 2000              # 화면 해상도 (픽셀 수)
x = np.linspace(-screen_width/2, screen_width/2, N)   # 화면 좌표

# --------------------------------------------------------------
# 2) 슬릿 투과 함수 (두 개의 직사각형)
# --------------------------------------------------------------
def aperture(x):
    """두 개의 슬릿을 만든다. 1 = 투과, 0 = 차단"""
    slit1 = np.abs(x + d/2) < a/2   # 왼쪽 슬릿
    slit2 = np.abs(x - d/2) < a/2   # 오른쪽 슬릿
    return np.where(slit1 | slit2, 1.0, 0.0)

# 슬릿 함수 (1차원)
A = aperture(x)

# --------------------------------------------------------------
# 3) Fraunhofer 회절 (푸리에 변환) → 화면의 복소 파면
# --------------------------------------------------------------
# 파면은 1차원 푸리에 변환이므로 np.fft 사용
#   U(k) = FT{ A(x) * exp(i k0 x sinθ) }  (여기선 θ≈0)
#   k = 2π/λ * x/L   (far‑field 관계)
k = 2*np.pi/lam * x / L          # 화면 좌표와 대응되는 파수

# 푸리에 변환 (FFT) → 복소 진폭
U = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))   # 중앙을 0에 맞춤
U = U * np.exp(1j * np.angle(U))                     # 위상 보존 (필요시)

# 화면에서의 복소 파면 (스케일링)
U_screen = U * np.exp(1j * 2*np.pi/lam * L) / (1j*lam*L)

# 강도 (Intensity)
I = np.abs(U_screen)**2
I = I / I.max()          # 정규화 (0~1)

# --------------------------------------------------------------
# 4) 시각화 (정적 + 애니메이션)
# --------------------------------------------------------------
fig, (ax_slit, ax_int) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                      gridspec_kw={'height_ratios':[1,2]})

# (a) 슬릿 모양
ax_slit.plot(x*1e3, A, color='k')
ax_slit.set_ylabel('Aperture')
ax_slit.set_title('Double‑slit (a = {:.1f} µm, d = {:.1f} µm)'.format(a*1e6, d*1e6))
ax_slit.set_ylim(-0.2, 1.2)
ax_slit.grid(True)

# (b) 정적 강도 패턴
line_int, = ax_int.plot(x*1e3, I, color='C0')
ax_int.set_xlabel('Screen position x (mm)')
ax_int.set_ylabel('Normalized intensity')
ax_int.set_title('Fraunhofer diffraction pattern (λ = {:.0f} nm)'.format(lam*1e9))
ax_int.grid(True)

# --------------------------------------------------------------
# 5) 파동 전파 애니메이션 (시간에 따라 파동이 슬릿을 통과 → 화면에 도달)
# --------------------------------------------------------------
# 파동 전파를 2‑D (x, z) 로 간단히 구현 (축소된 파동면)
z_max = L
Nz = 400
z = np.linspace(0, z_max, Nz)

# 초기 파동 (평면파)
def wavefield(x, z, t):
    """단순히 exp(i(kz - ωt)) 로 전파되는 평면파 + 슬릿 마스크"""
    k0 = 2*np.pi/lam
    omega = 2*np.pi*3e8/lam
    phase = k0*z - omega*t
    return np.exp(1j*phase) * aperture(x)

# 애니메이션 함수
def animate(frame):
    t = frame * 1e-14   # 임의의 시간 스텝
    # 현재 z 위치 (슬릿 바로 뒤에서부터 화면까지)
    zi = z[frame]
    # 파동을 슬릿에 통과시킨 뒤, 프라운호퍼 근사로 화면까지 전파
    # 여기서는 간단히 화면 강도만 업데이트
    line_int.set_ydata(I * (1 - np.exp(- (zi/L)**2)))   # 전파 진행에 따라 강도 점점 나타남
    ax_int.set_title(f'Propagation z = {zi*1e2:.1f} cm')
    return line_int,

ani = animation.FuncAnimation(fig, animate, frames=Nz, interval=30, blit=True)

# --------------------------------------------------------------
# 6) 실행
# --------------------------------------------------------------
plt.tight_layout()
plt.show()
