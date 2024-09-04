import os
import shutil
from pathlib import Path
import subprocess
from typing import List

class WebFileCopier:
    """
    A class to find and copy HTML, CSS, and JavaScript files from a source directory to a destination directory while preserving the directory structure.

    Attributes:
        source_dir (str): The source directory to search for HTML, CSS, and JavaScript files.
        dest_dir (str): The destination directory where the files will be copied.

    Methods:
        find_files() -> List[str]:
            Recursively find all HTML, CSS, and JavaScript files in the source directory.

        copy_files() -> None:
            Copy all found files to the destination directory.
    """

    def __init__(self, source_dir: str, dest_dir: str) -> None:
        """
        Initialize the WebFileCopier with source and destination directories.

        Args:
            source_dir (str): The source directory to search for files.
            dest_dir (str): The destination directory where files will be copied.
        """
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        os.makedirs(self.dest_dir, exist_ok=True)

    def find_files(self) -> List[str]:
        """
        Recursively find all HTML, CSS, and JavaScript files in the source directory.

        Returns:
            List[str]: A list of paths to HTML, CSS, and JavaScript files.
        """
        file_types = ('.html', '.css', '.js', '.md')
        files = []
        for root, _, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.lower().endswith(file_types):
                    full_path = os.path.join(root, filename)
                    files.append(full_path)
                    print(f"Found file: {full_path}")
        return files

    def copy_files(self) -> None:
        """
        Copy all found HTML, CSS, and JavaScript files to the destination directory, preserving the directory structure.
        """
        files = self.find_files()
        for file in files:
            # Calculate relative path to preserve directory structure
            relative_path = os.path.relpath(file, self.source_dir)
            dest_path = os.path.join(self.dest_dir, relative_path)
            dest_dir = os.path.dirname(dest_path)

            # Ensure the destination directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                print(f"Created directory: {dest_dir}")

            # Copy the file
            shutil.copy2(file, dest_path)
            print(f"Copied file from {file} to {dest_path}")

def cp_files(source_directory = '/Users/mx/Documents/0-GitHub/CLASS/Recommendation-System-Tutorial') -> None:
    """
    Main function to demonstrate the usage of WebFileCopier.
    """
    # destination_directory = './Recommendation-System-Tutorial'
    script_dir = Path(__file__).parent.resolve()
    destination_folder = 'BOOK'
    destination_directory = script_dir / destination_folder
    # 结合目标目录的完整路径
    destination_directory = destination_directory.resolve()
    destination_directory = os.path.join(destination_directory, source_directory.split('/')[-1]) 
    copier = WebFileCopier(source_directory, destination_directory)
    copier.copy_files()

        
class FileManager:
    """文件管理器，用于复制CSS和JS文件并执行命令"""
    
    def __init__(self, source_dir: str, destination_dir: str) -> None:
        """
        初始化文件管理器
        
        :param source_dir: 源目录
        :param destination_dir: 目标目录
        """
        self.source_dir = Path(source_dir)
        self.destination_dir = Path(destination_dir)
        self._validate_directories()

    def _validate_directories(self) -> None:
        """验证源目录和目标目录是否存在"""
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"源目录 {self.source_dir} 不存在。")
        if not self.destination_dir.is_dir():
            raise FileNotFoundError(f"目标目录 {self.destination_dir} 不存在。")

    def copy_files(self, extensions: List[str]) -> None:
        """
        复制指定扩展名的文件到目标目录
        
        :param extensions: 要复制的文件扩展名列表，例如 ['.css', '.js']
        """
        for ext in extensions:
            for file in self.source_dir.rglob(f"*{ext}"):
                destination_file = self.destination_dir / file.name
                shutil.copy(file, destination_file)
                print(f"复制文件: {file} 到 {destination_file}")

    def run_command(self, command: str) -> None:
        """
        在目标目录中运行指定命令
        
        :param command: 要执行的命令
        """
        os.chdir(self.destination_dir)
        print(f"切换到目标目录: {self.destination_dir}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"命令成功执行: {command}")
        else:
            print(f"命令执行失败: {command}")
            print(f"错误信息: {result.stderr}")

def build_html(source_directory='/Users/mx/Documents/0-GitHub/0_USEFULL_TOOLS/',destination_directory='/Users/mx/Documents/0-GitHub/CLASS/The_Elements_of_Statistical_Learning/'):
    
    file_manager = FileManager(source_directory, destination_directory)
    
    # 复制CSS和JS文件
    file_manager.copy_files(extensions=['.css', '.js'])
    
    # 运行Node.js脚本
    file_manager.run_command('node convert.js')
    file_manager.run_command('rm -rf convert.js')

if __name__ == '__main__':
    parent_path = '/Users/mx/Documents/0-GitHub/CLASS'
    source_directories = [os.path.join(parent_path, source_directory) for source_directory in os.listdir(parent_path) if not source_directory.startswith('.')]
    for source_directory in source_directories:
        if not os.path.exists(os.path.join(source_directory, 'convert.js')):
            print(f"Building HTML for {source_directory}...")
            build_html(destination_directory=source_directory)
            print(f'copying files for {source_directory}...')
            cp_files(source_directory=source_directory)
        else:
            print(f"Skipping {source_directory} as it does not contain an index.html file.")