%% captureIQ.m
% =========================================================================
%  [第1段] USRP B205 mini-i で Wi-Fi の生 IQ をキャプチャして HDD に保存する
% -------------------------------------------------------------------------
%  概要:
%    Wi-Fi の受信処理は「取得」と「復号」に分かれており、本スクリプトは
%    そのうちの取得のみを担当する。復号は decodeIQ_HE.m で行う。
%
%    分離している理由:
%      USRP は 20 MSps で連続的にサンプルを送出し続けるため、受信ループの
%      中で復号処理を挟むと処理が追いつかずオーバーラン (サンプルの
%      取りこぼし) が発生する。実測で 1 秒のキャプチャに対し復号は 100 秒
%      以上かかるので、リアルタイム処理は成立しない。そこで本スクリプトは
%      「受け取ってメモリに積むだけ」に徹する。
%
%  処理の流れ:
%    1. USRP を開いて指定時間だけ IQ サンプルを受信する
%    2. 複素 double 形式で HDD に .mat 保存する
%
%  必要環境:
%    - MATLAB / Communications Toolbox
%    - Communications Toolbox Support Package for USRP Radio
%    - USB 3.0 接続された USRP B205 mini-i
%    - ホストPC に接続された HDD (空き容量に注意。下記参照)
%
%  出力ファイル:
%    <hddSavePath>/<saveFileName>_raw.mat   (例: D1_E_antenna1_raw.mat)
%    saveFileName が '' なら <hddSavePath>/<mmddHHMM>_raw.mat (mmddHHMM = 月日時分)
%      iq   … 受信 IQ サンプル (complex double 列ベクトル)
%      meta … 中心周波数・サンプルレート・オーバーラン回数等の取得条件
%
%    ※ADC は 12bit、USB 経由のサンプル形式は 16bit であり、single (仮数
%      24bit) でも情報は失われないが、後段の decodeIQ_HE.m / WLAN Toolbox が
%      double 前提であるため、型変換を挟まず double のまま保存する。
%
%    ※ファイルサイズの目安 (20 MSps, complex double = 16 byte/sample):
%        1 秒 → 約 320 MB,  10 秒 → 約 3.2 GB,  90 秒 → 約 28.8 GB
%      キャプチャ中は同じサイズをメモリ上にも確保するため、実際の上限は
%      HDD 容量ではなく空きメモリ量で決まる点に注意。
%
%  次の手順:
%    このファイルを decodeIQ_HE.m の入力として指定する (既定では最新の
%    *_raw.mat が自動選択されるため、通常は指定不要)。
%
%  注意:
%    電波の受信・記録は、利用地域の電波法および関連法令を遵守し、
%    自身が管理する機器・許可された環境でのみ実施してください。
% =========================================================================

%% ------------------------------------------------------------------------
%  0. 直前の実行で残っている USRP オブジェクトの解放
%  ------------------------------------------------------------------------
% スクリプトの変数はベースワークスペースに残るため、前回がエラーで終了して
% いると受信機オブジェクトが USRP を掴んだままになり、次回実行時に
% "radio is busy" となる。clear より先に明示的に release する。
if exist('rx', 'var')
    try
        release(rx);
    catch
    end
end
clear; clc;

%% ------------------------------------------------------------------------
%  1. ユーザ設定パラメータ
%  ------------------------------------------------------------------------

% --- 保存先 (ホストPC に接続された HDD) ---------------------------------
%     現在の環境: HDPC-UT (D:)
hddSavePath = 'D:\IQ_raw';

% --- 保存ファイル名 ------------------------------------------------------
%     hddSavePath に <saveFileName>_raw.mat という名前で保存する。
%     データセット・活動・アンテナごとにここを書き換える
%       例: 'D1_E_antenna1'  ->  D1_E_antenna1_raw.mat
%     この名前は復号結果にも引き継がれる (D1_E_antenna1_WAX202_CSI.mat)。
%     '' にすると、従来どおり収集を始めた時刻の名前 (mmddHHMM) になる。
%     同じ名前のファイルが既にある場合は、上書きしないよう収集の前に止まる。
saveFileName = 'D1_E_antenna1';

% --- Wi-Fi 受信パラメータ (5GHz 帯 ch36 / 20MHz) -------------------------
centerFrequency = 5.180e9;      % [Hz] 5GHz 帯 ch36 (20MHz 帯域幅)
sampleRate      = 20e6;         % [Sps] 20 MHz 帯域
gain            = 30;           % [dB] B200 系は 0〜76 dB 程度
                                 %      飽和するなら下げ、弱すぎるなら上げる
captureDuration = 90.0;          % [s] Beacon 間隔は通常 100ms なので
                                 %     確実に捕捉したい場合は長めに設定。
                                 %     90s で約 28.8 GB (メモリも同量必要)。
                                 %     ※decodeIQ_HE.m の復号はキャプチャ長の
                                 %       40〜60 倍程度かかる (5s で 3〜5 分、
                                 %       電波の混雑度に依存)。長時間の
                                 %       キャプチャは復号時間に注意。
samplesPerFrame = 20000;        % [samples/frame]
usrpPlatform    = 'B200';
usrpSerialNum   = '32275DE';

%% ------------------------------------------------------------------------
%  2. 保存先の準備
%  ------------------------------------------------------------------------
if ~exist(hddSavePath, 'dir')
    fprintf('保存先フォルダが存在しないため作成します: %s\n', hddSavePath);
    [ok, msg] = mkdir(hddSavePath);
    if ~ok
        error('captureIQ:mkdirFailed', ...
            'HDD 保存先フォルダを作成できませんでした (%s): %s', hddSavePath, msg);
    end
end

testFile = fullfile(hddSavePath, '.write_test.tmp');
fidTest  = fopen(testFile, 'w');
if fidTest == -1
    error('captureIQ:hddNotWritable', ...
        'HDD 保存先に書き込みできません: %s', hddSavePath);
end
fclose(fidTest);
delete(testFile);

% --- 保存ファイル名の確認 ---
% 収集を始める前に確かめる (長い収集が終わってから上書きに気付くと
% 取り直しになるため)。'_raw' や '.mat' まで書いてあっても受け付ける。
saveName = char(regexprep(strtrim(char(saveFileName)), '(_raw)?(\.mat)?$', ''));
if ~isempty(saveName)
    if ~isempty(regexp(saveName, '[\\/:*?"<>|]', 'once'))
        error('captureIQ:badFileName', ...
            'saveFileName にファイル名に使えない文字が入っています: %s', saveName);
    end
    rawMatFile = fullfile(hddSavePath, [saveName '_raw.mat']);
    if isfile(rawMatFile)
        error('captureIQ:fileExists', ...
            ['同じ名前のファイルが既にあります: %s\n', ...
             '上書きしないよう止めました。saveFileName を変えるか、', ...
             '既にあるファイルを移動してください。'], rawMatFile);
    end
end

% 保存サイズの事前表示 (complex double で 16 byte/sample)
% 同じサイズをキャプチャ中にメモリ上へも確保するため、空きメモリも要確認。
estimatedBytes = captureDuration * sampleRate * 16;
fprintf('保存予定サイズ: 約 %.2f GB (メモリも同量必要)\n', estimatedBytes / 1e9);

% saveFileName が '' のときは、ファイル名に時刻 mmddHHMM (月日時分) を使う。
% 時刻には年が入らないので、
% 年まで含む完全な日時は meta.captureDatetimeFull に別途記録する。
timestamp  = datestr(now, 'mmddHHMM');
if isempty(saveName)
    rawMatFile = fullfile(hddSavePath, [timestamp '_raw.mat']);
end

%% ------------------------------------------------------------------------
%  3. USRP B205 mini-i 受信機オブジェクトの生成
%  ------------------------------------------------------------------------
radioInfo = [];
try
    radioInfo = findsdru();
    if isempty(radioInfo)
        warning('captureIQ:noRadio', ...
            'USRP 機器が検出されませんでした。USB 接続と電源を確認してください。');
    else
        fprintf('検出された USRP 機器:\n');
        for k = 1:numel(radioInfo)
            fprintf('  Platform=%s, SerialNum=%s, Status=%s\n', ...
                radioInfo(k).Platform, radioInfo(k).SerialNum, radioInfo(k).Status);
        end
    end
catch ME
    warning('captureIQ:findsdruFailed', 'findsdru の実行に失敗しました: %s', ME.message);
end

if isempty(usrpSerialNum)
    error('captureIQ:noSerialNum', ...
        'usrpSerialNum を指定してください (findsdru の表示を参照)。');
end

rx = comm.SDRuReceiver( ...
    'Platform',            usrpPlatform, ...
    'SerialNum',           usrpSerialNum, ...
    'CenterFrequency',     centerFrequency, ...
    'Gain',                gain, ...
    'MasterClockRate',     sampleRate * 2, ...
    'DecimationFactor',    2, ...
    'OutputDataType',      'double', ...   % WLAN Toolbox 関数は double 前提
    'SamplesPerFrame',     samplesPerFrame);

fprintf('\n受信設定:\n');
fprintf('  中心周波数      : %.4f GHz\n', centerFrequency / 1e9);
fprintf('  サンプルレート  : %.3f MSps (帯域幅)\n', sampleRate / 1e6);
fprintf('  ゲイン          : %d dB\n', gain);
fprintf('  キャプチャ時間  : %.2f s\n', captureDuration);
fprintf('  保存先(.mat)    : %s\n', rawMatFile);

%% ------------------------------------------------------------------------
%  4. IQ キャプチャ (復号は行わない)
%  ------------------------------------------------------------------------
totalSamplesTarget = round(captureDuration * sampleRate);
iqBuffer = complex(zeros(totalSamplesTarget + samplesPerFrame, 1));
totalSamplesCaptured = 0;
overrunCount = 0;

fprintf('\nIQ キャプチャを開始します...\n');
captureTic = tic;

% エラーや中断が起きても USRP を必ず解放する
% (スクリプトの onCleanup はベースワークスペースに残り発火しないため、
%  try/catch で明示的に解放する)
try
    while totalSamplesCaptured < totalSamplesTarget
        [iqData, dataLen, overrun] = rx();
        if dataLen == 0
            continue;
        end
        if overrun
            overrunCount = overrunCount + 1;
        end
        iqBuffer(totalSamplesCaptured + (1:dataLen)) = iqData(1:dataLen);
        totalSamplesCaptured = totalSamplesCaptured + dataLen;
    end
catch captureErr
    try
        release(rx);
    catch
    end
    rethrow(captureErr);
end

elapsedCapture = toc(captureTic);
release(rx);

iq = iqBuffer(1:totalSamplesCaptured);
clear iqBuffer;

fprintf('  キャプチャ完了: %d サンプル, %.2f s, オーバーラン %d 回\n', ...
    totalSamplesCaptured, elapsedCapture, overrunCount);

%% ------------------------------------------------------------------------
%  5. IQ とメタデータを HDD へ保存
%  ------------------------------------------------------------------------
% 復号 (decodeIQ_HE.m) 側の WLAN Toolbox が double を要求するため、型変換を
% 挟まず complex double のまま保存する。
meta = struct();
meta.description      = 'USRP B205 mini-i captured Wi-Fi IQ samples (5GHz ch36)';
meta.dataFormat       = 'complex double column vector, variable name: iq';
meta.rawMatFile       = rawMatFile;
meta.saveName         = saveName;               % saveFileName ('' = 時刻の名前)
meta.wifiChannel      = 36;
meta.centerFrequency  = centerFrequency;        % [Hz]
meta.sampleRate       = sampleRate;             % [Sps] = 帯域幅
meta.bandwidth        = 20e6;                   % [Hz]
meta.gain             = gain;                   % [dB]
meta.platform         = usrpPlatform;
meta.serialNum        = usrpSerialNum;
meta.captureDuration  = captureDuration;        % [s]
meta.samplesPerFrame  = samplesPerFrame;
meta.totalSamples     = totalSamplesCaptured;
meta.overrunCount     = overrunCount;
meta.elapsedCapture   = elapsedCapture;         % [s] 実際に要した時間
meta.captureDatetime     = timestamp;   % 'mmddHHMM' (収集を始めた時刻)
meta.captureDatetimeFull = datestr(now, 'yyyy-mm-dd HH:MM:SS');   % 年を含む完全な日時
meta.matlabVersion    = version;

% IQ が 2GB を超え得るため -v7.3 (HDF5 ベース) で保存する
save(rawMatFile, 'iq', 'meta', '-v7.3');

fprintf('\n生IQを保存しました: %s\n', rawMatFile);
if overrunCount > 0
    fprintf(['※オーバーランが %d 回発生しました。サンプルの取りこぼしが\n', ...
             '  あるため、samplesPerFrame を増やす、USB3.0 ポートを使う、\n', ...
             '  他の負荷の高いアプリを終了する等を検討してください。\n'], overrunCount);
end
fprintf('次に decodeIQ_HE.m を実行してください (このファイルが自動選択されます)。\n');
