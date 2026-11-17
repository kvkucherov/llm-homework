"""Install the pinned official Battlesnake CLI locally, verifying its SHA-256."""
import hashlib
import io
import platform
import tarfile
import urllib.request
from pathlib import Path

VERSION="1.2.3"


def main():
    system=platform.system()
    machine={"aarch64":"arm64","arm64":"arm64","x86_64":"x86_64","AMD64":"x86_64"}.get(platform.machine())
    if system not in ("Darwin","Linux") or not machine:
        raise SystemExit("Use the official release page for your platform: https://github.com/BattlesnakeOfficial/rules/releases")
    name=f"battlesnake_{VERSION}_{system}_{machine}.tar.gz"
    base=f"https://github.com/BattlesnakeOfficial/rules/releases/download/v{VERSION}/"
    def download(name):
        with urllib.request.urlopen(base+name,timeout=60) as response:return response.read()
    checksums=download("checksums.txt").decode()
    expected=next(line.split()[0] for line in checksums.splitlines() if line.split()[-1]==name)
    data=download(name)
    if hashlib.sha256(data).hexdigest()!=expected:raise SystemExit("Checksum mismatch")
    root=Path("vendor/battlesnake");root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as tar:
        for filename in ("battlesnake","LICENSE","README.md"):
            member=tar.getmember(filename)
            if not member.isfile():raise SystemExit("Unexpected archive entry")
            (root/filename).write_bytes(tar.extractfile(member).read())
    (root/"battlesnake").chmod(0o755)
    print(f"Installed official CLI v{VERSION}: {root/'battlesnake'}")


if __name__=="__main__":main()
