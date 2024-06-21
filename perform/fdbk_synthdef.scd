// Synthdef


(
SynthDef.new(\fdbk, {
	|
	fund = 440,
	add_amp = 0.5,
	blip_amp = 0.5,
	fm_amp = 0.5,
	ratio = 0.1,
	index = 1,
	pnoise_amp = 0.5,
	bnoise_amp = 0.5,
	lpf_freq = 5000,
	hpf_freq = 100,

	pan = 0,
	input = 0,

	env_amount = 1,
	lag_control = 1,
	delay_control = 1
	|

	var lagt, delayTime;
	var blip, global_amp, osc, fm, mod, pink, brown, dry_sig, wet_sig, sig, filt, rev, out;

	lagt = MouseX.kr(0, 3, lag: 0.1) * lag_control;
	delayTime = MouseY.kr(0.05, 3) * delay_control;


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

	global_amp = Amplitude.kr(DelayN.ar(SoundIn.ar(input), maxdelaytime: 10, delaytime: delayTime/2)).lag(0.01) * env_amount;
	sig = sig * (global_amp + 1 - env_amount);

	rev = FreeVerb.ar(sig, lagt/3.0);
	out = Out.ar(0, Pan2.ar(rev, pan))

}).add
);



