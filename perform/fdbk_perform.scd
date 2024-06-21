s.boot

s.

(SynthDef.new(\fdbk, {
	|
	fund = 440,
	add_amp = 1,
	blip_amp = 1,
	fm_amp = 1,
	ratio = 0.1,
	index = 1,
	pnoise_amp = 1,
	bnoise_amp = 1,
	lpf_freq = 5000,
	hpf_freq = 100,
	env_amount = 1,
	pan = 0,
	input = 0
	|

	var blip, global_amp, osc, fm, mod, pink, brown, dry_sig, wet_sig, sig, filt, rev, out;
	var lagt, delayTime;

	lagt = MouseX.kr(0, 3, lag: 0.1);
	delayTime = MouseY.kr(0.05, 3);

	//fund = DelayN.kr(fundam, maxdelaytime:1, delaytime:1)

	blip = Blip.ar(fund*0.5, 50, mul: blip_amp);

	osc = DynKlang.ar(`[
		[fund.lag(lagt),
			fund.lag(lagt)*2,
			fund.lag(lagt)*3,
			fund.lag(lagt)*4,
			fund.lag(lagt)*5,
			fund.lag(lagt)*6
		],
		[add_amp.lag(lagt),
			add_amp.lag(lagt)*0.5,
			add_amp.lag(lagt)*0.25,
			add_amp.lag(lagt)*0.2,
			add_amp.lag(lagt)*0.15,
			add_amp.lag(lagt)*0.15
		],
		nil],
	1, 0);

	mod = SinOsc.ar(fund.lag(lagt) * ratio.lag(lagt), 0, mul: index.lag(lagt) * fund).lag(lagt);
	fm = SinOsc.ar(fund.lag(lagt) + mod, mul: fm_amp.lag(lagt));

	pink = PinkNoise.ar(mul: pnoise_amp.lag(lagt));
	brown = BrownNoise.ar(mul: bnoise_amp.lag(lagt));

	dry_sig = osc + fm + pink;
	wet_sig = blip + brown;

	filt = LPF.ar(wet_sig, lpf_freq.lag(lagt));
	filt = HPF.ar(filt, hpf_freq.lag(lagt));

	sig = dry_sig + filt;
	//rev = sig;

	global_amp = Amplitude.kr(DelayN.ar(SoundIn.ar(input), maxdelaytime: 10, delaytime: delayTime/2), releaseTime: 0.01).lag(0.01);
	// global_amp = EnvFollow.kr(SoundIn.ar(input)).lag(lagt);
	//global_amp.postln;

	sig = sig * global_amp; //(rev * (1-env_amount)) + (rev * global_amp);  //* -30.dbamp
	rev = FreeVerb.ar(sig, lagt/3.0);

	//sig = DelayN.ar(rev, maxdelaytime: 10, delaytime: 0);


	out = Out.ar(0, Pan2.ar(rev, pan))
}).add;);



~leftSynth = Synth.new(\fdbk, [\pan, -1, \input, 0]);

~rightSynth = Synth.new(\fdbk, [\pan, 1, \input, 1]);



(OSCdef('\channel0', {
	arg msg;

	msg.postln;

	~leftSynth.set(
		\fund, msg[1],
	    \add_amp, msg[2],
		\blip_amp, msg[3],
		\fm_amp, msg[4],
		\ratio, msg[5],
		\index, msg[6],
		\pnoise_amp, msg[7],
		\bnoise_amp, msg[8],
		\lfp_freq, msg[9],
	    \hfp_freq, msg[10]);
}, '/channel_0'));

(OSCdef('\channel1', {
	arg msg;

	msg.postln;

	~rightSynth.set(
		\fund, msg[1],
	    \add_amp, msg[2],
		\blip_amp, msg[3],
		\fm_amp, msg[4],
		\ratio, msg[5],
		\index, msg[6],
		\pnoise_amp, msg[7],
		\bnoise_amp, msg[8],
		\lfp_freq, msg[9],
	    \hfp_freq, msg[10]);
}, '/channel_1'))




{MouseY.kr.poll}.play


/*
x = Synth.new(\fdbk_b);
x.set(\dist, 0.9, \fund, 1000)
*/



