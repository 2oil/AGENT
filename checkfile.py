#!/usr/bin/env python3
"""
FLAC 파일 손상 분석 도구
"""

import os
import sys
import struct
import subprocess
import soundfile as sf
from pathlib import Path

def analyze_file_corruption(filepath):
    """파일 손상 상태를 다각도로 분석"""
    filename = os.path.basename(filepath)
    print(f"\n🔍 Analyzing corruption: {filename}")
    print("=" * 80)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
    
    # 1. 기본 파일 정보
    file_size = os.path.getsize(filepath)
    print(f"📁 File size: {file_size:,} bytes")
    
    # 2. FLAC 헤더 분석
    print(f"\n🔍 FLAC Header Analysis:")
    try:
        with open(filepath, 'rb') as f:
            # FLAC signature 확인
            signature = f.read(4)
            if signature == b'fLaC':
                print(f"✅ FLAC signature valid: {signature}")
            else:
                print(f"❌ Invalid FLAC signature: {signature}")
                print(f"   Expected: b'fLaC', Got: {signature}")
            
            # 메타데이터 블록들 읽기
            block_count = 0
            while block_count < 10:  # 안전장치
                block_header = f.read(4)
                if len(block_header) < 4:
                    break
                    
                # 블록 헤더 파싱
                last_block = (block_header[0] & 0x80) != 0
                block_type = block_header[0] & 0x7F
                block_size = int.from_bytes(block_header[1:4], 'big')
                
                block_types = {
                    0: "STREAMINFO",
                    1: "PADDING", 
                    2: "APPLICATION",
                    3: "SEEKTABLE",
                    4: "VORBIS_COMMENT",
                    5: "CUESHEET",
                    6: "PICTURE"
                }
                
                block_name = block_types.get(block_type, f"UNKNOWN({block_type})")
                print(f"   Block {block_count}: {block_name}, size={block_size}, last={last_block}")
                
                # 블록 데이터 건너뛰기
                f.seek(block_size, 1)
                block_count += 1
                
                if last_block:
                    break
            
            # 오디오 프레임 시작 위치
            audio_start = f.tell()
            print(f"🎵 Audio frames start at byte: {audio_start}")
            
    except Exception as e:
        print(f"❌ Header analysis failed: {e}")
    
    # 3. soundfile 상세 오류 분석
    print(f"\n🔍 SoundFile Analysis:")
    try:
        info = sf.info(filepath)
        print(f"✅ Metadata readable:")
        print(f"   Duration: {info.frames / info.samplerate:.2f}s")
        print(f"   Sample rate: {info.samplerate} Hz")
        print(f"   Channels: {info.channels}")
        print(f"   Format: {info.format}, {info.subtype}")
        print(f"   Frames: {info.frames:,}")
    except Exception as e:
        print(f"❌ Metadata read failed: {e}")
    
    # 4. 부분적 읽기 테스트
    print(f"\n🔍 Partial Read Test:")
    chunk_sizes = [1000, 10000, 100000, 500000]  # 다양한 크기로 테스트
    
    for chunk_size in chunk_sizes:
        try:
            data, sr = sf.read(filepath, frames=chunk_size)
            print(f"✅ Read {chunk_size} frames: shape={data.shape}")
        except Exception as e:
            print(f"❌ Failed to read {chunk_size} frames: {e}")
            break
    
    # 5. 바이너리 패턴 분석
    print(f"\n🔍 Binary Pattern Analysis:")
    try:
        with open(filepath, 'rb') as f:
            # 파일의 여러 지점 샘플링
            file_size = os.path.getsize(filepath)
            sample_points = [0, file_size//4, file_size//2, file_size*3//4, file_size-100]
            
            for i, pos in enumerate(sample_points):
                if pos >= file_size:
                    continue
                f.seek(pos)
                sample = f.read(16)
                hex_str = ' '.join(f'{b:02x}' for b in sample)
                print(f"   Pos {pos:,}: {hex_str}")
                
                # 이상한 패턴 체크
                if len(set(sample)) == 1:  # 모든 바이트가 같음
                    print(f"     ⚠️ Suspicious: All bytes are 0x{sample[0]:02x}")
                
    except Exception as e:
        print(f"❌ Binary analysis failed: {e}")
    
    # 6. 외부 도구로 검증 (ffmpeg가 있다면)
    print(f"\n🔍 External Tool Validation:")
    try:
        # ffprobe로 검증
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', filepath
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ ffprobe validation passed")
            # JSON 파싱해서 주요 정보 출력할 수도 있음
        else:
            print(f"❌ ffprobe validation failed:")
            print(f"   stderr: {result.stderr}")
            
    except FileNotFoundError:
        print(f"⚠️ ffprobe not available (install ffmpeg for more detailed analysis)")
    except subprocess.TimeoutExpired:
        print(f"❌ ffprobe timeout")
    except Exception as e:
        print(f"❌ ffprobe error: {e}")
    
    # 7. 파일 무결성 추정
    print(f"\n📊 Corruption Assessment:")
    try:
        info = sf.info(filepath)
        expected_size = info.frames * info.channels * 2  # 16-bit PCM 추정
        actual_size = file_size
        
        print(f"   Expected raw size: ~{expected_size:,} bytes")
        print(f"   Actual file size: {actual_size:,} bytes")
        print(f"   Compression ratio: {actual_size/expected_size:.2f}")
        
        if actual_size < expected_size * 0.1:  # 너무 작음
            print(f"   🔴 File seems severely truncated")
        elif actual_size < expected_size * 0.5:  # 작음
            print(f"   🟡 File might be partially corrupted")
        else:
            print(f"   🟢 File size seems reasonable")
            
    except:
        print(f"   ❌ Cannot assess file integrity")

def compare_with_good_file(corrupted_path, good_path):
    """정상 파일과 비교 분석"""
    print(f"\n🔍 Comparing with good file:")
    print("=" * 80)
    
    if not os.path.exists(good_path):
        print(f"❌ Good file not found: {good_path}")
        return
    
    # 헤더 비교
    try:
        with open(corrupted_path, 'rb') as cf, open(good_path, 'rb') as gf:
            corrupted_header = cf.read(1024)
            good_header = gf.read(1024)
            
            # 바이트별 비교
            diff_count = 0
            for i, (c, g) in enumerate(zip(corrupted_header, good_header)):
                if c != g:
                    diff_count += 1
                    if diff_count <= 10:  # 처음 10개 차이점만 출력
                        print(f"   Diff at byte {i}: corrupted=0x{c:02x}, good=0x{g:02x}")
            
            if diff_count == 0:
                print(f"✅ Headers identical in first 1024 bytes")
            else:
                print(f"❌ Found {diff_count} differences in first 1024 bytes")
                
    except Exception as e:
        print(f"❌ Header comparison failed: {e}")

def main():
    print("🧪 FLAC File Corruption Analysis Tool")
    print("=" * 80)
    
    corrupted_files = [
        'LA_0073_LA_D_1225110.flac',
        'LA_0070_LA_D_2172149.flac', 
        'LA_0069_LA_D_1600475.flac',
        'LA_0076_LA_D_1339494.flac',
        'LA_0075_LA_D_1146342.flac',
        'LA_0071_LA_D_1305336.flac',
        'LA_0077_LA_D_1921351.flac',
    ]
    
    base_dir = "./enr_audio/dev/"
    
    # 정상 파일 하나 찾기 (비교용)
    good_file = None
    for file in os.listdir(base_dir):
        if file.endswith('.flac') and file not in corrupted_files:
            good_file = os.path.join(base_dir, file)
            break
    
    if good_file:
        print(f"📋 Using as reference: {os.path.basename(good_file)}")
        analyze_file_corruption(good_file)
    
    # 손상된 파일들 분석
    for filename in corrupted_files[:3]:  # 처음 3개만 상세 분석
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            analyze_file_corruption(filepath)
            
            if good_file:
                compare_with_good_file(filepath, good_file)
        else:
            print(f"\n❌ File not found: {filename}")

if __name__ == "__main__":
    main()