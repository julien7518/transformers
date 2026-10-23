import React from 'react';
import {render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {beforeEach,expect,it,vi} from 'vitest';
import Translator from '@/components/translator';
import {characterCount,validateInput} from '@/lib/contracts';
const health = {status:'ready',model_id:'test-model',limits:{max_input_chars:500,max_source_tokens:64,max_new_tokens:63}};
beforeEach(()=>{vi.stubGlobal('fetch',vi.fn().mockResolvedValue(Response.json(health)));});
it('compte des points de code Unicode et refuse les entrées invalides',()=>{expect(characterCount('🙂é')).toBe(2);expect(validateInput('  ')).toBeTruthy();expect(validateInput('🙂'.repeat(500))).toBeNull();expect(validateInput('🙂'.repeat(501))).toBeTruthy();});
it('valide le texte vide puis traduit et inverse avec le résultat',async()=>{
 const user=userEvent.setup();render(<Translator mock={false}/>);
 await user.click(screen.getByRole('button',{name:'Changer de ton'}));expect(screen.getByText('Écris une phrase avant de la traduire.')).toBeInTheDocument();
 await user.type(screen.getByLabelText('Français standard'),'Bonjour');
 vi.mocked(fetch).mockResolvedValueOnce(Response.json({translation:'Salut',direction:'to_genz',truncated:true,model_id:'test-model'}));
 await user.click(screen.getByRole('button',{name:'Changer de ton'}));
 expect(await screen.findByText('Salut')).toBeInTheDocument();expect(screen.getByText(/La phrase peut être incomplète/)).toBeInTheDocument();
 await user.click(screen.getByRole('button',{name:'Inverser le sens de traduction'}));expect(screen.getByLabelText('Français Gen Z')).toHaveValue('Salut');expect(screen.queryByRole('button',{name:/Copier/})).not.toBeInTheDocument();
});
it('ignore une réponse devenue obsolète après modification et bloque la double soumission',async()=>{
 const user=userEvent.setup();render(<Translator mock={false}/>);await waitFor(()=>expect(fetch).toHaveBeenCalledTimes(1));
 let resolve!:(value:Response)=>void;vi.mocked(fetch).mockReturnValueOnce(new Promise(r=>{resolve=r;}));
 await user.type(screen.getByLabelText('Français standard'),'Bonjour');await user.click(screen.getByRole('button',{name:'Changer de ton'}));expect(screen.getByRole('button',{name:'Traduction en cours'})).toBeDisabled();
 await user.type(screen.getByLabelText('Français standard'),' !');resolve(Response.json({translation:'Réponse obsolète',direction:'to_genz',truncated:false,model_id:'test'}));
 await waitFor(()=>expect(screen.queryByText('Réponse obsolète')).not.toBeInTheDocument());expect(screen.getByRole('button',{name:'Changer de ton'})).toBeEnabled();
});
it('affiche une erreur backend et permet de réessayer',async()=>{
 const user=userEvent.setup();render(<Translator mock={true}/>);expect(screen.getByText(/réponses simulées/)).toBeInTheDocument();
 await user.click(screen.getByRole('button',{name:/Ce film est vraiment excellent/}));
 vi.mocked(fetch).mockResolvedValueOnce(Response.json({error:{code:'MODEL_UNAVAILABLE',message:'Aucun modèle entraîné n’est disponible.'}},{status:503}));
 await user.click(screen.getByRole('button',{name:'Changer de ton'}));expect(await screen.findByText('Aucun modèle entraîné n’est disponible.')).toBeInTheDocument();expect(screen.getByRole('button',{name:'Changer de ton'})).toBeEnabled();
});
it('copie le résultat et confirme',async()=>{
 const user=userEvent.setup();render(<Translator mock={false}/>);await user.type(screen.getByLabelText('Français standard'),'Bonjour');
 vi.mocked(fetch).mockResolvedValueOnce(Response.json({translation:'Salut',direction:'to_genz',truncated:false,model_id:'test'}));await user.click(screen.getByRole('button',{name:'Changer de ton'}));await user.click(await screen.findByRole('button',{name:/Copier/}));expect(await screen.findByText('Traduction copiée.')).toBeInTheDocument();expect(await navigator.clipboard.readText()).toBe('Salut');
});
