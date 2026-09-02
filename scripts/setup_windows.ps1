<#
.SYNOPSIS
    Monta el entorno completo de LeRobot en Windows 11, con los arreglos que
    hicieron falta de verdad en esta maquina.

.DESCRIPTION
    Es la version ejecutable de docs/00-instalacion-windows.md. Cada paso
    corresponde a un problema documentado alli.

    Lo que NO hace, a proposito:
      - No crea la carpeta raiz ni concede permisos: eso necesita elevacion y
        prefiero que lo hagas tu conscientemente (ver -Root abajo).
      - No activa el Modo Desarrollador de Windows: tambien necesita elevacion.
        El script comprueba si esta activo y te dice el comando exacto si no.

.PARAMETER Root
    Carpeta donde va todo lo pesado (venv, caches, datasets, checkpoints).
    Tiene que existir y ser escribible SIN elevar. Si esta en la raiz de un
    disco secundario, probablemente no lo sea: ver docs/00, Problema 2.

.PARAMETER Python312
    Ruta al python.exe 3.12. Localizalo con `py -0p`.

.EXAMPLE
    .\setup_windows.ps1 -Root D:\robotics-lab
#>

[CmdletBinding()]
param(
    [string]$Root = "D:\robotics-lab",
    [string]$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    OK   $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "    AVISO $msg" -ForegroundColor Yellow }
function Die($msg)      { Write-Host "    ERROR $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 0. checks --

Step 0 "Comprobaciones previas"

if (-not (Test-Path $Python312)) {
    Die "No encuentro Python 3.12 en $Python312. Localizalo con 'py -0p' y pasalo con -Python312.`n" +
        "         lerobot exige >=3.12, y 3.13/3.14 suele ser demasiado nuevo para las dependencias."
}
$pyver = & $Python312 --version
Ok "$pyver en $Python312"

if (-not (Test-Path $Root)) {
    Die @"
La carpeta $Root no existe.
         La raiz de un disco secundario en Windows solo deja escribir a
         administradores, asi que crearla y darte permiso necesita elevacion.
         Desde una consola COMO ADMINISTRADOR, una sola vez:

             New-Item -ItemType Directory "$Root"
             icacls "$Root" /grant "`$env:USERNAME:(OI)(CI)F"

         El icacls es imprescindible: la carpeta hereda permisos de solo
         lectura del disco y sin el seguirias sin poder escribir dentro.
"@
}

$probe = Join-Path $Root ".__wtest"
try {
    New-Item -ItemType File $probe -ErrorAction Stop | Out-Null
    Remove-Item $probe -Force
    Ok "$Root existe y es escribible sin elevar"
} catch {
    Die "$Root existe pero NO es escribible. Te falta el icacls /grant de arriba."
}

$free = (Get-PSDrive ($Root[0])).Free / 1GB
if ($free -lt 20) { Warn ("solo {0:N1} GB libres en el disco de destino" -f $free) }
else { Ok ("{0:N1} GB libres en el disco de destino" -f $free) }

# Modo Desarrollador: sin el, lerobot revienta al guardar checkpoints y
# huggingface_hub falla al bajar algunos datasets (ver docs/00, Problema 2
# y docs/03, tropiezo 3).
$devmode = 0
try {
    $devmode = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" `
                -Name AllowDevelopmentWithoutDevLicense -ErrorAction Stop).AllowDevelopmentWithoutDevLicense
} catch { }

if ($devmode -eq 1) {
    Ok "Modo Desarrollador activo (los symlinks funcionaran)"
} else {
    Warn @"
Modo Desarrollador NO activo.
          lerobot crea un symlink 'checkpoints/last' al guardar y fallara con
          [WinError 1314] DESPUES de haber guardado el checkpoint, que es
          especialmente confuso. Desde una consola COMO ADMINISTRADOR:

              reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v AllowDevelopmentWithoutDevLicense /d 1

          Sigo, pero el entrenamiento fallara al primer checkpoint.
"@
}

# ------------------------------------------------------------------ 1. venv --

Step 1 "Entorno virtual"

$venv = Join-Path $Root "venv"
$py   = Join-Path $venv "Scripts\python.exe"

if (Test-Path $py) {
    Ok "ya existe en $venv"
} else {
    # Un venv guarda rutas absolutas: no se puede mover, hay que recrearlo.
    & $Python312 -m venv $venv
    Ok "creado en $venv"
}
& $py -m pip install --upgrade pip --quiet
Ok "pip actualizado"

# ---------------------------------------------------------------- 2. caches --

Step 2 "Redirigir caches fuera del disco de sistema"

foreach ($d in @("cache\huggingface", "cache\pip", "cache\torch", "tmp")) {
    $p = Join-Path $Root $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory $p -Force | Out-Null }
}

$env:HF_HOME       = Join-Path $Root "cache\huggingface"
$env:PIP_CACHE_DIR = Join-Path $Root "cache\pip"
$env:TORCH_HOME    = Join-Path $Root "cache\torch"
# La critica: pip descomprime los wheels aqui, y el de torch pesa ~2.5 GB.
$env:TMP           = Join-Path $Root "tmp"
$env:TEMP          = $env:TMP
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Ok "HF_HOME, PIP_CACHE_DIR, TORCH_HOME y TMP/TEMP apuntan a $Root"

# ----------------------------------------------------------------- 3. torch --

Step 3 "PyTorch con CUDA (~2.5 GB, tarda)"

$hasTorch = & $py -c "import torch,sys; sys.stdout.write(torch.__version__)" 2>$null
if ($LASTEXITCODE -eq 0 -and $hasTorch) {
    Ok "ya instalado: $hasTorch"
} else {
    # Instalar torch ANTES que lerobot para elegir nosotros la build de CUDA;
    # si no, pip resuelve la version de CPU desde PyPI.
    & $py -m pip install "torch==2.11.*" torchvision --index-url $CudaIndex
    if ($LASTEXITCODE -ne 0) { Die "fallo instalando torch" }
    Ok "instalado"
}

# --------------------------------------------------------------- 4. lerobot --

Step 4 "LeRobot en modo editable"

$lerobot = Join-Path $Root "lerobot"
if (-not (Test-Path $lerobot)) {
    git clone --depth 1 https://github.com/huggingface/lerobot.git $lerobot
    Ok "clonado"
} else {
    Ok "ya clonado en $lerobot"
}

Push-Location $lerobot
try {
    & $py -m pip install -e ".[dataset,training,pusht,diffusion,hardware,viz]"
    if ($LASTEXITCODE -ne 0) { Die "fallo instalando lerobot" }
} finally { Pop-Location }
Ok "lerobot instalado"

# ----------------------------------------------------------------- 5. ffmpeg --

Step 5 "FFmpeg 8.x (shared) para torchcodec"

$ffdir = Join-Path $Root "ffmpeg"
if (Test-Path (Join-Path $ffdir "bin\ffmpeg.exe")) {
    Ok "ya instalado en $ffdir"
} else {
    # DOS trampas aqui, ver docs/00 Problema 7:
    #  - el build 'master-latest' que todo el mundo enlaza es FFmpeg 9-dev y
    #    torchcodec solo soporta 4-8. Hay que coger una release etiquetada.
    #  - tiene que ser 'shared' (con DLLs sueltas), no 'static'.
    $url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip"
    $zip = Join-Path $env:TMP "ffmpeg81.zip"
    $ext = Join-Path $env:TMP "ff81"

    Write-Host "    descargando FFmpeg 8.1 shared..."
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $ext -Force
    Move-Item (Get-ChildItem $ext -Directory | Select-Object -First 1).FullName $ffdir -Force
    Remove-Item $zip -Force; Remove-Item $ext -Recurse -Force -ErrorAction SilentlyContinue
    Ok "instalado en $ffdir"
}

# ---------------------------------------------------------- 6. sitecustomize --

Step 6 "Registrar las DLLs de FFmpeg para torchcodec"

# Anadir FFmpeg al PATH NO funciona: desde Python 3.8 Windows ignora PATH para
# resolver dependencias de modulos de extension. Hace falta os.add_dll_directory,
# y sitecustomize.py es la forma de aplicarlo a todo proceso del venv.
$src = Join-Path $PSScriptRoot "sitecustomize.py"
$dst = Join-Path $venv "Lib\site-packages\sitecustomize.py"
if (-not (Test-Path $src)) { Die "no encuentro $src" }

(Get-Content $src -Raw) -replace 'Path\(r"D:\\robotics-lab\\ffmpeg\\bin"\)', "Path(r`"$ffdir\bin`")" |
    Set-Content $dst -Encoding utf8
Ok "sitecustomize.py instalado apuntando a $ffdir\bin"

# ------------------------------------------------------------------ 7. env.ps1 --

Step 7 "Generar env.ps1"

$envFile = Join-Path $Root "env.ps1"
@"
# Generado por scripts/setup_windows.ps1
# Uso:  . $envFile

`$Root = "$Root"

`$env:HF_HOME       = "`$Root\cache\huggingface"
`$env:PIP_CACHE_DIR = "`$Root\cache\pip"
`$env:TORCH_HOME    = "`$Root\cache\torch"
# CRITICO: pip descomprime wheels aqui; el de torch pesa ~2.5 GB.
`$env:TMP           = "`$Root\tmp"
`$env:TEMP          = "`$Root\tmp"

# huggingface_hub deduplica con symlinks; sin privilegio unos datasets bajan
# bien y otros petan con [WinError 1314]. Forzamos copias.
`$env:HF_HUB_DISABLE_SYMLINKS = "1"
`$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

`$env:PATH = "`$Root\ffmpeg\bin;" + `$env:PATH

# Llamamos al interprete por ruta absoluta en vez de "activar" el venv, asi
# cada comando es reproducible y no depende del estado de la shell.
`$PY = "`$Root\venv\Scripts\python.exe"

Write-Host "[env] PY=`$PY" -ForegroundColor DarkGray
"@ | Set-Content $envFile -Encoding utf8
Ok "escrito en $envFile"

# ------------------------------------------------------------ 8. verificacion --

Step 8 "Verificacion"

$check = @'
import importlib, sys
fallos = []
for m in ["lerobot","torch","torchvision","torchcodec","cv2","av","datasets",
          "gymnasium","gym_pusht","diffusers","serial","rerun"]:
    try:
        mod = importlib.import_module(m)
        print(f"    OK    {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"    FALLA {m:14s} {type(e).__name__}: {str(e)[:80]}")
        fallos.append(m)
import torch
print(f"    CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
else:
    print("    AVISO: sin CUDA, el entrenamiento ira por CPU (muy lento)")
sys.exit(1 if fallos else 0)
'@
$env:PATH = "$ffdir\bin;" + $env:PATH
& $py -c $check
$verifyOk = ($LASTEXITCODE -eq 0)

Write-Host ""
if ($verifyOk) {
    Write-Host "Entorno listo. Empieza con:" -ForegroundColor Green
    Write-Host "    . $envFile"
    Write-Host "    & `$PY scripts\explore_dataset.py lerobot/pusht"
} else {
    Write-Host "Hay imports que fallan. Mira docs/00-instalacion-windows.md." -ForegroundColor Red
    exit 1
}
