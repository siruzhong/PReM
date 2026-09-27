import ast
import shlex
import subprocess
import sys
import unittest
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/eval/run_eval_online.py"


def declared_cli_options(script: Path) -> set[str]:
    tree = ast.parse(script.read_text())
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        options.update(
            arg.value for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
        )
    return options


class OnlineEvalCommandContractTest(unittest.TestCase):

    def dry_run(self, mode: str) -> list[str]:
        command = [
            sys.executable,
            str(RUNNER),
            "--mode",
            mode,
            "--scenario",
            "public",
            "--model_path",
            "model",
            "--output_dir",
            "outputs",
            "--evaluation_name",
            "contract",
            "--cuda_devices",
            "0,1",
            "--dry_run",
        ]
        if mode == "prem":
            command.extend(["--prem_ckpt", "prem.pt"])
        command.extend(["--datasets", "longvideobench", "--workers_per_gpu", "1"])

        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.startswith("[exec] ")]
        self.assertEqual(len(lines), 1, completed.stdout)
        return shlex.split(lines[0][len("[exec] "):])

    def assert_common_contract(self, command: list[str], mode: str):
        engine = {
            "prem": "eval_prem_online.py",
            "flash": "eval_flash_vstream_online.py",
            "qwen2vl": "eval_qwen2vl_online.py",
            "qwen3vl": "eval_qwen2vl_online.py",
        }[mode]
        self.assertTrue(command[1].endswith(engine))
        generated_options = {part for part in command[2:] if part.startswith("--")}
        accepted_options = declared_cli_options(REPO_ROOT / "scripts/eval" / engine)
        self.assertFalse(
            generated_options - accepted_options,
            f"{engine} does not accept {sorted(generated_options - accepted_options)}",
        )
        self.assertEqual(command[command.index("--num_chunks") + 1], "2")
        self.assertEqual(command[command.index("--dataset") + 1], "longvideobench")
        self.assertTrue(command[command.index("--gt_file") + 1].endswith("lvb_val.json"))

    def test_public_commands_match_each_engine_cli(self):
        for mode in ("prem", "flash", "qwen2vl", "qwen3vl"):
            with self.subTest(mode=mode):
                command = self.dry_run(mode)
                self.assert_common_contract(command, mode)
                if mode == "prem":
                    self.assertIn("--prem_ckpt", command)
                    self.assertIn("--prem_alpha", command)
                    self.assertNotIn("--prem_max_memory_tokens", command)
                    self.assertIn("--max_new_tokens", command)
                elif mode == "flash":
                    self.assertIn("--qwen_model", command)
                    self.assertIn("--initial_chunk_frames", command)
                    self.assertIn("--chunk_frames", command)
                    self.assertNotIn("--max_new_tokens", command)
                else:
                    self.assertIn("--buffer_frames", command)
                    self.assertIn("--max_video_tokens", command)
                    self.assertNotIn("--max_new_tokens", command)


if __name__ == "__main__":
    unittest.main()
