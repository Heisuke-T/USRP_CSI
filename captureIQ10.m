%% captureIQ10.m
% =========================================================================
%  [第1段] 実行してから 10 秒後に IQ キャプチャを始める captureIQ.m
% -------------------------------------------------------------------------
%  目的:
%    別の場所・別の PC に置いた 2 台の USRP で同時に収集するとき、
%    収集開始の時刻をなるべく揃えるためのもの。captureIQ.m との違いは
%    「いつ収集を始めるか」だけで、出力の形式は同じ (decodeIQ_HE.m で
%    そのまま復号できる)。
%
%  動き方:
%    実行ボタンを押した瞬間を基準時刻 (t=0) として記録し、
%      t=0 〜        USRP の検出・受信オブジェクト生成・初回の受信
%                    (FPGA の読み込み)・バッファ確保 … ここまでが準備
%      準備完了 〜 t=10  受信は続けたまま、届いたサンプルを読み捨てる
%      t=10         ここから記録を開始する
%    という順で進む。
%
%    単純に冒頭で pause(10) すると、その後の準備に PC ごとに違う時間
%    (数秒〜十数秒) がかかるので、開始時刻がばらつく。準備を待ち時間の
%    中に収めることで、開始時刻を「実行から 10 秒」に揃えている
%    (誤差は 1 フレーム = 約 1 ms 程度)。
%
%    待ち時間の間も受信を止めずに読み捨てているのは、止めて待つと
%    USB 側に古いサンプルが溜まり、開始直後のデータが「過去のもの」に
%    なったりオーバーランになったりするため。
%
%  揃う精度について (重要):
%    この方法で揃うのは「各 PC で実行ボタンを押した時刻」が揃っている
%    範囲まで。2 人で声を掛け合って押すなら数百 ms〜1 秒程度ずれる。
%    そのため、後から精密に合わせられるよう次の値を meta に記録する。
%      meta.captureStartWallClock  … 収集を始めた PC の時計の時刻 (ms まで)
%      meta.captureStartPosix      … 同じ時刻の UNIX 時間 [s] (UTC 基準)
%    2 台の PC の時計が NTP で合っていれば、この差でおおよそ揃えられる。
%    最終的な精密な位置合わせは、両方の受信機が受けた同じビーコンの
%    TSF タイムスタンプで行うのが確実 (復号側で対応予定)。
%
%  出力ファイル:
%    <hddSavePath>/<saveFileName>_raw.mat   (例: D1_E_antenna1_raw.mat)
%      iq   … 受信 IQ サンプル (complex double 列ベクトル)
%      meta … 取得条件 + 上記の開始時刻の記録
%    名前は復号結果 (*_CSI.mat) にも引き継がれる。2 台で取るときは、
%    PC ごとにアンテナの部分 (antenna1 / antenna2) を変えておく。
%    saveFileName が '' なら <mmddHHMM>_<シリアル番号>_raw.mat (2 台分を
%    同じフォルダに集めても衝突しないよう、USRP のシリアル番号を入れる)。
%
%  2 台で使うときの手順の例:
%    1. 両方の PC で、このファイルの保存先・ゲイン・キャプチャ時間を揃え、
%       saveFileName のアンテナの部分だけを PC ごとに変える
%    2. 合図で同時に実行する (またはどちらかが captureIQ10.m、もう一方が
%       captureIQ15.m を 5 秒遅れで実行する、など)
%    3. 表示される「収集開始時刻」を両方メモしておく
%
%  必要環境・ファイルサイズ・注意事項は captureIQ.m と同じ。
% =========================================================================

%% ------------------------------------------------------------------------
%  0. 実行した瞬間の記録と、直前の実行で残っている USRP の解放
%  ------------------------------------------------------------------------
% 実行ボタンを押した時刻を、ほかの何よりも先に記録する。
% 収集はここから startDelay 秒後に始める。
scriptStartTic  = tic;
scriptStartWall = datetime('now', 'TimeZone', 'local');

% スクリプトの変数はベースワークスペースに残るため、前回がエラーで終了して
% いると受信機オブジェクトが USRP を掴んだままになり、次回実行時に
% "radio is busy" となる。clear より先に明示的に release する。
if exist('rx', 'var')
    try
        release(rx);
    catch
    end
end
clearvars -except scriptStartTic scriptStartWall
clc;

%% ------------------------------------------------------------------------
%  1. ユーザ設定パラメータ
%  ------------------------------------------------------------------------

% --- 収集開始までの時間 --------------------------------------------------
startDelay = 10.0;             % [s] 実行ボタンを押してから収集を始めるまで
                                 %     USRP の準備はこの時間内に済ませる。
                                 %     準備がこれより長くかかった場合は、
                                 %     警告を出して準備完了と同時に始める。

% --- 保存先 (ホストPC に接続された HDD) ---------------------------------
%     現在の環境: HDPC-UT (D:)
hddSavePath = 'D:\IQ_raw';

% --- 保存ファイル名 ------------------------------------------------------
%     hddSavePath に <saveFileName>_raw.mat という名前で保存する。
%     データセット・活動・アンテナごとにここを書き換える
%       例: 'D1_E_antenna1'  ->  D1_E_antenna1_raw.mat
%     この名前は復号結果にも引き継がれる (D1_E_antenna1_WAX202_CSI.mat)。
%     '' にすると、従来どおり収集を始めた時刻の名前 (mmddHHMM_<シリアル番号>) になる。
%     同じ名前のファイルが既にある場合は、上書きしないよう収集の前に止まる。
saveFileName = 'D1_E_antenna1';

% --- Wi-Fi 受信パラメータ (5GHz 帯 ch36 / 20MHz) -------------------------
%     2 台で収集するときは、両方の PC でここを揃えること。
centerFrequency = 5.180e9;      % [Hz] 5GHz 帯 ch36 (20MHz 帯域幅)
sampleRate      = 20e6;         % [Sps] 20 MHz 帯域
gain            = 30;           % [dB] B200 系は 0〜76 dB 程度
                                 %      飽和するなら下げ、弱すぎるなら上げる
captureDuration = 90.0;         % [s] キャプチャ時間。complex double で
                                 %     1 秒あたり 320 MB のメモリを使う
                                 %     (90 秒で 28.8 GB)。
samplesPerFrame = 20000;        % [samples/frame]
usrpPlatform    = 'B200';

%     '' のままにすると、つながっている USRP が 1 台ならそれを自動で使う。
%     2 台を別々の PC で使う場合、PC ごとにシリアル番号を書き換えずに
%     同じファイルを使えるようにするため。1 台の PC に複数つないでいる
%     ときは、使う方のシリアル番号を書くこと (findsdru の表示を参照)。
usrpSerialNum   = '';

%% ------------------------------------------------------------------------
%  2. 保存先の準備
%  ------------------------------------------------------------------------
if ~exist(hddSavePath, 'dir')
    fprintf('保存先フォルダが存在しないため作成します: %s\n', hddSavePath);
    [ok, msg] = mkdir(hddSavePath);
    if ~ok
        error('captureIQ10:mkdirFailed', ...
            'HDD 保存先フォルダを作成できませんでした (%s): %s', hddSavePath, msg);
    end
end

testFile = fullfile(hddSavePath, '.write_test.tmp');
fidTest  = fopen(testFile, 'w');
if fidTest == -1
    error('captureIQ10:hddNotWritable', ...
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
        error('captureIQ10:badFileName', ...
            'saveFileName にファイル名に使えない文字が入っています: %s', saveName);
    end
    rawMatFile = fullfile(hddSavePath, [saveName '_raw.mat']);
    if isfile(rawMatFile)
        error('captureIQ10:fileExists', ...
            ['同じ名前のファイルが既にあります: %s\n', ...
             '上書きしないよう止めました。saveFileName を変えるか、', ...
             '既にあるファイルを移動してください。'], rawMatFile);
    end
end

% 保存サイズの事前表示 (complex double で 16 byte/sample)
% 同じサイズをキャプチャ中にメモリ上へも確保するため、空きメモリも要確認。
estimatedBytes = captureDuration * sampleRate * 16;
fprintf('保存予定サイズ: 約 %.2f GB (メモリも同量必要)\n', estimatedBytes / 1e9);

%% ------------------------------------------------------------------------
%  3. USRP B205 mini-i 受信機オブジェクトの生成
%  ------------------------------------------------------------------------
radioInfo = [];
try
    radioInfo = findsdru();
    if isempty(radioInfo)
        warning('captureIQ10:noRadio', ...
            'USRP 機器が検出されませんでした。USB 接続と電源を確認してください。');
    else
        fprintf('検出された USRP 機器:\n');
        for k = 1:numel(radioInfo)
            fprintf('  Platform=%s, SerialNum=%s, Status=%s\n', ...
                radioInfo(k).Platform, radioInfo(k).SerialNum, radioInfo(k).Status);
        end
    end
catch ME
    warning('captureIQ10:findsdruFailed', 'findsdru の実行に失敗しました: %s', ME.message);
end

if isempty(usrpSerialNum)
    if numel(radioInfo) == 1
        usrpSerialNum = char(string(radioInfo(1).SerialNum));
        fprintf('usrpSerialNum が空なので、検出された 1 台を使います: %s\n', usrpSerialNum);
    elseif isempty(radioInfo)
        error('captureIQ10:noSerialNum', ...
            ['USRP が見つからないため、使う機器を決められません。\n', ...
             'USB 接続と電源を確認するか、usrpSerialNum を指定してください。']);
    else
        error('captureIQ10:multipleRadios', ...
            ['USRP が %d 台つながっています。使う方のシリアル番号を\n', ...
             'usrpSerialNum に指定してください (上の一覧を参照)。'], numel(radioInfo));
    end
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
fprintf('  USRP シリアル   : %s\n', usrpSerialNum);
fprintf('  保存先フォルダ  : %s\n', hddSavePath);
if isempty(saveName)
    fprintf('  保存ファイル名  : (収集開始の時刻)_%s_raw.mat\n', ...
        regexprep(usrpSerialNum, '[^A-Za-z0-9]', ''));
else
    fprintf('  保存ファイル名  : %s_raw.mat\n', saveName);
end
fprintf('  実行した時刻    : %s\n', ...
    fmtTime(scriptStartWall, 'yyyy-MM-dd HH:mm:ss.SSS'));
fprintf('  収集開始の予定  : 実行から %.1f 秒後\n', startDelay);

%% ------------------------------------------------------------------------
%  4. IQ キャプチャ (開始時刻まで待ってから記録する)
%  ------------------------------------------------------------------------
totalSamplesTarget = round(captureDuration * sampleRate);
iqBuffer = complex(zeros(totalSamplesTarget + samplesPerFrame, 1));
totalSamplesCaptured = 0;
overrunCount = 0;
missedStartDeadline = false;

% エラーや中断が起きても USRP を必ず解放する
% (スクリプトの onCleanup はベースワークスペースに残り発火しないため、
%  try/catch で明示的に解放する)
try
    % --- 受信の立ち上げ ---
    % 初回の呼び出しでハードウェアが初期化される (FPGA の読み込み等で
    % 数秒かかることがある)。ここも待ち時間の中に収める。
    fprintf('\nUSRP を初期化しています...\n');
    rx();
    initDoneSec = toc(scriptStartTic);
    fprintf('  準備完了 (実行から %.1f 秒)\n', initDoneSec);

    if initDoneSec > startDelay
        % 準備が間に合わなかった。待たずにすぐ始め、そのことを記録する。
        missedStartDeadline = true;
        warning('captureIQ10:missedStartDeadline', ...
            ['準備に %.1f 秒かかり、指定の %.1f 秒に間に合いませんでした。\n', ...
             'すぐに収集を始めます。開始時刻は meta に記録されるので、\n', ...
             '後から位置合わせはできます。毎回間に合わない場合は\n', ...
             'startDelay を長くしてください。'], initDoneSec, startDelay);
    else
        % --- 開始時刻まで、受信を続けたまま読み捨てる ---
        lastShown = -1;
        while true
            remaining = startDelay - toc(scriptStartTic);
            if remaining <= 0
                break;
            end
            sec = ceil(remaining);
            if sec ~= lastShown
                fprintf('  収集開始まで %d 秒\n', sec);
                lastShown = sec;
            end
            rx();   % 読み捨て
        end
    end

    % --- ここが収集開始の瞬間 ---
    actualStartDelay = toc(scriptStartTic);
    captureStartWall = datetime('now', 'TimeZone', 'local');
    captureTic = tic;
    fprintf('\nIQ キャプチャを開始しました: %s (実行から %.3f 秒)\n', ...
        fmtTime(captureStartWall, 'yyyy-MM-dd HH:mm:ss.SSS'), ...
        actualStartDelay);

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
captureEndWall = datetime('now', 'TimeZone', 'local');
release(rx);

iq = iqBuffer(1:totalSamplesCaptured);
clear iqBuffer;

fprintf('  キャプチャ完了: %d サンプル, %.2f s, オーバーラン %d 回\n', ...
    totalSamplesCaptured, elapsedCapture, overrunCount);

%% ------------------------------------------------------------------------
%  5. IQ とメタデータを HDD へ保存
%  ------------------------------------------------------------------------
% saveFileName を指定していればその名前 (上で決めた rawMatFile)。
% '' のときは「収集を始めた時刻 (mmddHHMM)」+「USRP のシリアル番号」。
% 2 台分を同じフォルダに集めても衝突しないようにシリアル番号を入れる。
% 時刻 + シリアル番号は meta.captureDatetime として復号側にも引き継がれる。
serialTag  = regexprep(usrpSerialNum, '[^A-Za-z0-9]', '');
timestamp  = [fmtTime(captureStartWall, 'MMddHHmm') '_' serialTag];
if isempty(saveName)
    rawMatFile = fullfile(hddSavePath, [timestamp '_raw.mat']);
end

% 復号 (decodeIQ_HE.m) 側の WLAN Toolbox が double を要求するため、型変換を
% 挟まず complex double のまま保存する。
meta = struct();
meta.description      = 'USRP B205 mini-i captured Wi-Fi IQ samples (5GHz ch36), delayed start';
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
meta.hostName         = getenv('COMPUTERNAME'); % どの PC で取ったか (Windows)
meta.captureDuration  = captureDuration;        % [s]
meta.samplesPerFrame  = samplesPerFrame;
meta.totalSamples     = totalSamplesCaptured;
meta.overrunCount     = overrunCount;
meta.elapsedCapture   = elapsedCapture;         % [s] 実際に要した時間
meta.captureDatetime     = timestamp;   % 'mmddHHMM_<シリアル>' (収集開始の時刻)
meta.captureDatetimeFull = fmtTime(captureStartWall, 'yyyy-MM-dd HH:mm:ss');
meta.matlabVersion    = version;

% --- 2 台の位置合わせ用の時刻記録 ---
% WallClock は人が読む用 (PC のローカル時刻、ms まで)。
% Posix は計算用 (1970-01-01 UTC からの秒数)。2 台の差はこちらで取る。
meta.startDelay            = startDelay;          % [s] 指定した待ち時間
meta.actualStartDelay      = actualStartDelay;    % [s] 実際に始まった時刻 (実行から)
meta.missedStartDeadline   = missedStartDeadline; % true = 準備が間に合わなかった
meta.scriptStartWallClock  = fmtTime(scriptStartWall, 'yyyy-MM-dd HH:mm:ss.SSS');
meta.captureStartWallClock = fmtTime(captureStartWall, 'yyyy-MM-dd HH:mm:ss.SSS');
meta.captureEndWallClock   = fmtTime(captureEndWall, 'yyyy-MM-dd HH:mm:ss.SSS');
meta.scriptStartPosix      = posixtime(scriptStartWall);    % [s]
meta.captureStartPosix     = posixtime(captureStartWall);   % [s]
meta.captureEndPosix       = posixtime(captureEndWall);     % [s]

% IQ が 2GB を超え得るため -v7.3 (HDF5 ベース) で保存する
save(rawMatFile, 'iq', 'meta', '-v7.3');

fprintf('\n生IQを保存しました: %s\n', rawMatFile);
fprintf('  収集開始時刻 : %s\n', meta.captureStartWallClock);
fprintf('  (もう一方の PC の収集開始時刻と見比べてください)\n');
if missedStartDeadline
    fprintf(['※準備が間に合わず、指定より遅れて (実行から %.1f 秒で) 収集を\n', ...
             '  始めました。startDelay を長くすることを検討してください。\n'], ...
        actualStartDelay);
end
if overrunCount > 0
    fprintf(['※オーバーランが %d 回発生しました。サンプルの取りこぼしが\n', ...
             '  あるため、samplesPerFrame を増やす、USB3.0 ポートを使う、\n', ...
             '  他の負荷の高いアプリを終了する等を検討してください。\n'], overrunCount);
end
fprintf('次に decodeIQ_HE.m を実行してください (このファイルが自動選択されます)。\n');

%% ------------------------------------------------------------------------
%  ローカル関数
%  ------------------------------------------------------------------------
function s = fmtTime(t, fmt)
    % datetime を指定の書式の文字列にする (元の変数の書式は変えない)
    t.Format = fmt;
    s = char(t);
end
