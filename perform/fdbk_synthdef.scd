
SynthDef.new(\fdbk, {
	|
	fund = 440,
	amp_ratio = 2,
	fm_ratio = 0.1,
	fm_index = 1,
	pnoise_amp = 0.5,
	lpf_freq = 5000,
	hpf_freq = 100,
	osc_fm_ratio = 0.5,

	in_chan = 0,
	dry_env = 1,
	global_amp = 1,
	verb_mix = 0,
	out_chan = 0,

	mix = 0,
	lag_t = 0,
	delay_t = 0
	|

	var l_fund, l_amp_ratio, l_fm_ratio, l_fm_index, l_pnoise_amp, l_lpf_freq, l_hpf_freq, l_osc_fm_ratio;
	var harm, mod, fm, noise, lpf, hpf, sum, env_follow, sig, delayed_sig, verb, mixer, out;

	l_fund = fund.lag(lag_t);
	l_amp_ratio = amp_ratio.lag(lag_t);
	l_fm_ratio = fm_ratio.lag(lag_t);
	l_fm_index = fm_index.lag(lag_t);
	l_pnoise_amp = pnoise_amp.lag(lag_t);
	l_lpf_freq = lpf_freq.lag(lag_t);
	l_hpf_freq = hpf_freq.lag(lag_t);
	l_osc_fm_ratio = osc_fm_ratio.lag(lag_t);

	harm = DynKlang.ar(`[
		[l_fund,
			l_fund*2,
			l_fund*3,
			l_fund*4,
			l_fund*5,
			l_fund*6,
			l_fund*7,
			l_fund*8
		],
		[1,
			0.8**(l_amp_ratio*1.5),
			0.8**(l_amp_ratio*2),
			0.8**(l_amp_ratio*2.5),
			0.8**(l_amp_ratio*3),
			0.8**(l_amp_ratio*3.5),
			0.8**(l_amp_ratio*4),
			0.8**(l_amp_ratio*4.5)
		],
		nil],
	1, 0);

	mod = SinOsc.ar(l_fund * l_fm_ratio, 0, mul: l_fm_index * l_fund);
	fm = SinOsc.ar(l_fund + mod);

	noise = PinkNoise.ar(l_pnoise_amp);

	sum = (harm * l_osc_fm_ratio) + (fm * (1-l_osc_fm_ratio)) + noise;

	lpf = LPF.ar(sum, l_lpf_freq);
	hpf = HPF.ar(lpf, l_hpf_freq);

	env_follow = Amplitude.kr(SoundIn.ar(in_chan)).lag(0.01);
	sig = (hpf * dry_env * env_follow) + (hpf * (1 - dry_env * env_follow)) * global_amp;
	delayed_sig = DelayN.ar(sig, maxdelaytime: 10, delaytime: delay_t);
	verb = FreeVerb.ar(delayed_sig, verb_mix);

	mixer = XFade2.ar(SoundIn.ar(in_chan), verb, pan: mix);

	out = Out.ar(out_chan, mixer);
}).add;




	