import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------- Lectura y conversión a µs --------------------
valores = []
with open('muones.txt', 'r') as f:
    for linea in f:
        if not linea.strip():
            continue
        t_ns = float(linea.split()[0])      # 1ª col: tiempo (ns)
        if t_ns <= 10000:                   # mantener hasta 40,000 ns
            valores.append(t_ns / 1000.0)   # -> µs

valores.sort()
N_total = len(valores)
print(f"N_total (<= 10 µs): {N_total}")

# -------------------- Bins de 20 ns = 0.02 µs --------------------
bins = np.arange(0.0, 40.0 + 0.02, 0.02)    # [µs]
freq, edges = np.histogram(valores, bins=bins)

# -------------------- Frecuencia acumulada y sobrevivientes --------------------
freq_acum = np.cumsum(freq)
sobreviv  = N_total - freq_acum

# Límite superior de cada bin (0.02, 0.04, ..., 40.00 µs)
bin_tops = edges[1:]

# -------------------- Serie para ajuste: (x,y) --------------------
# Incluimos el punto inicial (t=0, N_total)
x = np.concatenate(([0.0], bin_tops))        # tiempo en µs
y = np.concatenate(([N_total], sobreviv))    # N(t)

# Errores para la gráfica (opcionales)
xerr = np.full_like(x, 0.01, dtype=float)    # mitad del bin: 0.01 µs
yerr = np.sqrt(np.maximum(y, 1.0))           # Poisson


# ========================== 2) AJUSTE 1: A exp(-t/τ) + B =======================
TAU0_GUESS_US = 2.2
def model_with_bg(t, A, tau, B):
    return A * np.exp(-t / tau) + B
p0 = [float(N_total), TAU0_GUESS_US, 0.0]
popt, pcov = curve_fit(model_with_bg, x, y, p0=p0, maxfev=10000)
A_opt, tau_opt, B_opt = popt
perr = np.sqrt(np.diag(pcov))

print("\n--- Ajuste 1 (con fondo) ---")
print(f"A       = {A_opt:.6g} ± {perr[0]:.3g}")
print(f"tau_obs = {tau_opt:.6g} ± {perr[1]:.3g} µs")
print(f"B       = {B_opt:.6g} ± {perr[2]:.3g}")

t_smooth = np.linspace(0.0, x.max(), 600)
y_fit    = model_with_bg(x, *popt)
y_smooth = model_with_bg(t_smooth, *popt)
res1     = y - y_fit

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', ms=3, capsize=2,
               elinewidth=0.7, label="Datos (sobrevivientes)")
ax[0].plot(t_smooth, y_smooth, lw=1.6, label="Ajuste: A·e^{-t/τ} + B")
ax[0].set_ylabel("N(t)")
ax[0].set_title("Ajuste exponencial con fondo (t en µs)")
ax[0].legend(loc="upper right"); ax[0].grid(True, alpha=0.3)

ax[1].scatter(x, res1, s=10, label="Residuales")
ax[1].axhline(0.0, ls="--", c="k")
ax[1].set_xlabel("t [µs]"); ax[1].set_ylabel("Residual")
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("Ajuste1.pdf")

# ========================== 3) AJUSTE 2: FONDO REMOVIDO ========================
N_clean = y - B_opt
mask    = N_clean > 0.0
t_pure  = x[mask]
N_pure  = N_clean[mask]
def model_pure(t, A, tau):
    return A * np.exp(-t / tau)

p0_pure = [float(max(N_pure[0], 1.0)), float(tau_opt)]
popt_pure, pcov_pure = curve_fit(model_pure, t_pure, N_pure, p0=p0_pure, maxfev=10000)
A_pure, tau_pure = popt_pure
perr_pure = np.sqrt(np.diag(pcov_pure))

print("\n--- Ajuste 2a (exponencial, fondo removido) ---")
print(f"A'   = {A_pure:.6g} ± {perr_pure[0]:.3g}")
print(f"tau' = {tau_pure:.6g} ± {perr_pure[1]:.3g} µs")

tt = np.linspace(0.0, t_pure.max(), 600)
y_fit_pure = model_pure(t_pure, *popt_pure)
res2 = N_pure - y_fit_pure

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].scatter(t_pure, N_pure, s=10, label="Datos (N−B)")
ax[0].plot(tt, model_pure(tt, *popt_pure), "r-", lw=1.5,
           label=f"Ajuste puro: τ'={tau_pure:.3f} µs")
ax[0].set_ylabel("N_clean(t)"); ax[0].set_title("Ajuste exponencial con fondo removido (B=0)")
ax[0].legend(); ax[0].grid(True, alpha=0.3)

ax[1].scatter(t_pure, res2, s=10, label="Residuales")
ax[1].axhline(0.0, ls="--", c="k")
ax[1].set_xlabel("t [µs]"); ax[1].set_ylabel("Residual")
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("Ajuste2.pdf")

# 5b) Lineal en el log: ln[(N-B)/N0] vs t
N0 = N_pure[0]
y_log = np.log(N_pure / N0)

# Pesos Poisson: Var[ln N] ≈ 1/N -> sigma ≈ 1/sqrt(N)
sigma_log = 1.0 / np.sqrt(N_pure)
w = 1.0 / sigma_log

(m, b), cov = np.polyfit(t_pure, y_log, 1, w=w, cov=True)
sigma_m = np.sqrt(cov[0, 0]); sigma_b = np.sqrt(cov[1, 1])

tau_lin = -1.0 / m
sigma_tau_lin = sigma_m / (m*m)

print("\n--- Ajuste 2b (lineal en log, fondo removido) ---")
print(f"m       = {m:.6g} ± {sigma_m:.3g}  [1/µs]")
print(f"b       = {b:.6g} ± {sigma_b:.3g}")
print(f"tau_lin = {tau_lin:.6g} ± {sigma_tau_lin:.3g} µs")
print(f"(comparación) tau_obs (ajuste 1) = {tau_opt:.6g} µs")

y_log_fit = m * t_pure + b
res_log   = y_log - y_log_fit

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].scatter(t_pure, y_log, s=10, label="Datos (log, N→N−B)")
ax[0].plot(t_pure, y_log_fit, "r-", lw=1.5,
           label=f"Ajuste: m={m:.3e} ⇒ τ={tau_lin:.3f} µs")
ax[0].set_ylabel(r"$\ln[(N-B)/N_0]$")
ax[0].set_title("Segundo ajuste (log) con fondo removido")
ax[0].legend(); ax[0].grid(True, alpha=0.3)

ax[1].scatter(t_pure, res_log, s=10, label="Residuales (log)")
ax[1].axhline(0.0, ls="--", c="k")
ax[1].set_xlabel("t [µs]"); ax[1].set_ylabel("Residual (log)")
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("Ajuste3.pdf")

# ========================== 4) G_F/(ħc)^3 Y ERROR ==============================
def gf_over_hbarc3_with_error(tau_pure, sigma_tau_us):
    hbar_GeV_s = 6.582119569e-25
    m_mu_GeV   = 0.1056583745
    tau_s = tau_pure * 1e-6
    GF = np.sqrt((192.0*np.pi**3 * hbar_GeV_s) / (tau_s * m_mu_GeV**5))
    sigma_GF = 0.5 * GF * (sigma_tau_us / tau_pure)
    return GF, sigma_GF

GF_val, GF_err = gf_over_hbarc3_with_error(tau_pure, perr[1])
print(f"\nG_F/(ħ c)^3 = {GF_val:.6g} ± {GF_err:.3g} GeV^-2   (PDG ≈ 1.166e-5)")

# ========================== 5) RAZÓN ρ Y ERROR =================================
TAU_MINUS_US, SIGMA_TAU_MINUS_US = 2.043, 0.003   
TAU_PLUS_US,  SIGMA_TAU_PLUS_US  = 2.211, 0.003   

def rho_with_error(tau_obs, sigma_obs, tau_minus, sigma_minus, tau_plus, sigma_plus):
    inv_obs = 1.0 / tau_obs
    inv_m   = 1.0 / tau_minus
    inv_p   = 1.0 / tau_plus
    rho = (inv_m - inv_obs) / (inv_obs - inv_p)
    den = (inv_obs - inv_p)
    dra = 1.0 / den
    drb = - (inv_m - inv_p) / (den**2)
    drc = (inv_m - inv_obs) / (den**2)
    sigma_a = sigma_minus / (tau_minus**2)
    sigma_b = sigma_obs   / (tau_obs**2)
    sigma_c = sigma_plus  / (tau_plus**2)
    sigma_rho = np.sqrt((dra*sigma_a)**2 + (drb*sigma_b)**2 + (drc*sigma_c)**2)
    return rho, sigma_rho
rho, sigma_rho = rho_with_error(
    tau_obs=tau_pure, sigma_obs=perr[1],
    tau_minus=TAU_MINUS_US, sigma_minus=SIGMA_TAU_MINUS_US,
    tau_plus=TAU_PLUS_US,   sigma_plus=SIGMA_TAU_PLUS_US
)
print(f"rho = N^+/N^- = {rho:.4f} ± {sigma_rho:.4f}")
